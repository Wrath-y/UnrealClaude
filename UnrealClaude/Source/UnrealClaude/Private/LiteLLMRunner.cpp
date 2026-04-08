// Copyright Natali Caggiano. All Rights Reserved.

#include "LiteLLMRunner.h"
#include "UnrealClaudeModule.h"
#include "Async/Async.h"
#include "Dom/JsonObject.h"
#include "Dom/JsonValue.h"
#include "Serialization/JsonReader.h"
#include "Serialization/JsonWriter.h"
#include "Serialization/JsonSerializer.h"
#include "Misc/FileHelper.h"
#include "Misc/Paths.h"
#include "HttpModule.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"

// ---------------------------------------------------------------------------
// Config loading
// ---------------------------------------------------------------------------

bool FLiteLLMRunner::TryLoadConfig(const FString& ConfigFilePath, FLiteLLMConfig& OutConfig)
{
	FString JsonText;
	if (!FFileHelper::LoadFileToString(JsonText, *ConfigFilePath))
	{
		return false;
	}

	TSharedPtr<FJsonObject> Root;
	TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(JsonText);
	if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
	{
		UE_LOG(LogUnrealClaude, Warning, TEXT("LiteLLM: failed to parse %s"), *ConfigFilePath);
		return false;
	}

	Root->TryGetStringField(TEXT("base_url"),   OutConfig.BaseUrl);
	Root->TryGetStringField(TEXT("auth_token"),  OutConfig.AuthToken);
	Root->TryGetStringField(TEXT("model"),       OutConfig.Model);

	// Trim trailing slash from base_url
	OutConfig.BaseUrl.RemoveFromEnd(TEXT("/"));

	if (!OutConfig.IsValid())
	{
		UE_LOG(LogUnrealClaude, Warning,
			TEXT("LiteLLM: config at %s is missing required fields (base_url / auth_token / model)"),
			*ConfigFilePath);
		return false;
	}

	UE_LOG(LogUnrealClaude, Log,
		TEXT("LiteLLM: loaded config — base_url=%s  model=%s"),
		*OutConfig.BaseUrl, *OutConfig.Model);
	return true;
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

FLiteLLMRunner::FLiteLLMRunner(const FLiteLLMConfig& InConfig)
	: LiteLLMConfig(InConfig)
{
}

FLiteLLMRunner::~FLiteLLMRunner()
{
	Cancel();
}

// ---------------------------------------------------------------------------
// IClaudeRunner — Cancel
// ---------------------------------------------------------------------------

void FLiteLLMRunner::Cancel()
{
	if (ActiveRequest.IsValid())
	{
		ActiveRequest->CancelRequest();
		ActiveRequest.Reset();
	}
	bIsExecuting = false;
}

// ---------------------------------------------------------------------------
// Request body builder
// ---------------------------------------------------------------------------

FString FLiteLLMRunner::BuildRequestBody(const FClaudeRequestConfig& Config) const
{
	// User message only — system goes as top-level field per Anthropic Messages API
	TArray<TSharedPtr<FJsonValue>> Messages;
	{
		TSharedPtr<FJsonObject> UserMsg = MakeShared<FJsonObject>();
		UserMsg->SetStringField(TEXT("role"),    TEXT("user"));
		UserMsg->SetStringField(TEXT("content"), Config.Prompt);
		Messages.Add(MakeShared<FJsonValueObject>(UserMsg));
	}

	// Root object
	TSharedPtr<FJsonObject> Root = MakeShared<FJsonObject>();
	Root->SetStringField(TEXT("model"),      LiteLLMConfig.Model);
	Root->SetArrayField (TEXT("messages"),   Messages);
	Root->SetNumberField(TEXT("max_tokens"), 8096);
	// stream=false for simplicity; the response arrives in one shot
	Root->SetBoolField  (TEXT("stream"),     false);

	// System prompt is a top-level field in Anthropic Messages API (not a message role)
	if (!Config.SystemPrompt.IsEmpty())
	{
		Root->SetStringField(TEXT("system"), Config.SystemPrompt);
	}

	FString Body;
	TSharedRef<TJsonWriter<>> Writer = TJsonWriterFactory<>::Create(&Body);
	FJsonSerializer::Serialize(Root.ToSharedRef(), Writer);
	return Body;
}

// ---------------------------------------------------------------------------
// Response parser
// ---------------------------------------------------------------------------

FString FLiteLLMRunner::ParseResponseBody(const FString& ResponseBody) const
{
	// Anthropic Messages API response shape:
	// {"content":[{"type":"text","text":"..."}],"role":"assistant",...}
	TSharedPtr<FJsonObject> Root;
	TSharedRef<TJsonReader<>> Reader = TJsonReaderFactory<>::Create(ResponseBody);
	if (!FJsonSerializer::Deserialize(Reader, Root) || !Root.IsValid())
	{
		UE_LOG(LogUnrealClaude, Warning, TEXT("LiteLLM: failed to parse response JSON: %s"), *ResponseBody);
		return FString();
	}

	// Check for API-level error
	const TSharedPtr<FJsonObject>* ErrorObj;
	if (Root->TryGetObjectField(TEXT("error"), ErrorObj))
	{
		FString ErrMsg;
		(*ErrorObj)->TryGetStringField(TEXT("message"), ErrMsg);
		UE_LOG(LogUnrealClaude, Warning, TEXT("LiteLLM: API error: %s"), *ErrMsg);
		return FString::Printf(TEXT("[API error] %s"), *ErrMsg);
	}

	const TArray<TSharedPtr<FJsonValue>>* ContentArr;
	if (!Root->TryGetArrayField(TEXT("content"), ContentArr) || ContentArr->Num() == 0)
	{
		UE_LOG(LogUnrealClaude, Warning, TEXT("LiteLLM: no content in response: %s"), *ResponseBody);
		return FString();
	}

	// Concatenate all text blocks
	FString Result;
	for (const TSharedPtr<FJsonValue>& Item : *ContentArr)
	{
		const TSharedPtr<FJsonObject>* Block;
		if (!Item->TryGetObject(Block)) { continue; }
		FString Type;
		(*Block)->TryGetStringField(TEXT("type"), Type);
		if (Type == TEXT("text"))
		{
			FString Text;
			(*Block)->TryGetStringField(TEXT("text"), Text);
			Result += Text;
		}
	}
	return Result;
}

// ---------------------------------------------------------------------------
// HTTP callback
// ---------------------------------------------------------------------------

void FLiteLLMRunner::OnHttpRequestComplete(
	FHttpRequestPtr Request,
	FHttpResponsePtr Response,
	bool bWasSuccessful)
{
	// We're on whichever thread the HTTP module uses; marshal to game thread
	FOnClaudeResponse OnComplete  = PendingOnComplete;
	FOnClaudeProgress OnProgress  = PendingOnProgress;

	PendingOnComplete.Unbind();
	PendingOnProgress.Unbind();
	ActiveRequest.Reset();
	bIsExecuting = false;

	if (!bWasSuccessful || !Response.IsValid())
	{
		const int32 FailCode = Response.IsValid() ? Response->GetResponseCode() : -1;
		const FString FailBody = Response.IsValid() ? Response->GetContentAsString() : TEXT("(no response)");
		UE_LOG(LogUnrealClaude, Error,
			TEXT("LiteLLM: HTTP request failed — bWasSuccessful=%d  ResponseValid=%d  HttpCode=%d  Body=%s"),
			(int32)bWasSuccessful, (int32)Response.IsValid(), FailCode, *FailBody);
		FString ErrMsg = FString::Printf(
			TEXT("[LiteLLM] HTTP request failed (bSuccess=%d, code=%d). See Output Log for details."),
			(int32)bWasSuccessful, FailCode);
		AsyncTask(ENamedThreads::GameThread, [OnComplete, ErrMsg]()
		{
			OnComplete.ExecuteIfBound(ErrMsg, false);
		});
		return;
	}

	const int32 HttpCode = Response->GetResponseCode();
	const FString ResponseBody = Response->GetContentAsString();

	if (HttpCode < 200 || HttpCode >= 300)
	{
		UE_LOG(LogUnrealClaude, Warning,
			TEXT("LiteLLM: HTTP %d — %s"), HttpCode, *ResponseBody);

		FString ErrMsg = FString::Printf(TEXT("[LiteLLM] HTTP %d error"), HttpCode);
		AsyncTask(ENamedThreads::GameThread, [OnComplete, ErrMsg]()
		{
			OnComplete.ExecuteIfBound(ErrMsg, false);
		});
		return;
	}

	FString Content = ParseResponseBody(ResponseBody);
	const bool bSuccess = !Content.IsEmpty();

	UE_LOG(LogUnrealClaude, Log,
		TEXT("LiteLLM: response received (%d chars)"), Content.Len());

	AsyncTask(ENamedThreads::GameThread, [OnComplete, OnProgress, Content, bSuccess]()
	{
		if (OnProgress.IsBound())
		{
			OnProgress.Execute(Content);
		}
		OnComplete.ExecuteIfBound(Content, bSuccess);
	});
}

// ---------------------------------------------------------------------------
// IClaudeRunner — ExecuteAsync
// ---------------------------------------------------------------------------

bool FLiteLLMRunner::ExecuteAsync(
	const FClaudeRequestConfig& Config,
	FOnClaudeResponse OnComplete,
	FOnClaudeProgress OnProgress)
{
	bool Expected = false;
	if (!bIsExecuting.CompareExchange(Expected, true))
	{
		UE_LOG(LogUnrealClaude, Warning, TEXT("LiteLLM: already executing a request"));
		return false;
	}

	if (!LiteLLMConfig.IsValid())
	{
		bIsExecuting = false;
		OnComplete.ExecuteIfBound(TEXT("[LiteLLM] invalid config — check litellm-config.json"), false);
		return false;
	}

	const FString Url = LiteLLMConfig.BaseUrl + TEXT("/v1/messages");
	const FString Body = BuildRequestBody(Config);

	UE_LOG(LogUnrealClaude, Log,
		TEXT("LiteLLM: POST %s  model=%s  prompt_len=%d\nRequest Body: %s"),
		*Url, *LiteLLMConfig.Model, Config.Prompt.Len(), *Body);

	TSharedRef<IHttpRequest, ESPMode::ThreadSafe> HttpRequest =
		FHttpModule::Get().CreateRequest();

	HttpRequest->SetURL(Url);
	HttpRequest->SetVerb(TEXT("POST"));
	HttpRequest->SetHeader(TEXT("Content-Type"),  TEXT("application/json"));
	HttpRequest->SetHeader(TEXT("Authorization"),
		FString::Printf(TEXT("Bearer %s"), *LiteLLMConfig.AuthToken));
	HttpRequest->SetContentAsString(Body);
	HttpRequest->SetTimeout(600.0f);  // 10-minute timeout for long model responses

	// Store callbacks before binding the delegate (lambda captures them)
	PendingOnComplete = OnComplete;
	PendingOnProgress = OnProgress;
	ActiveRequest = HttpRequest;

	HttpRequest->OnProcessRequestComplete().BindRaw(
		this, &FLiteLLMRunner::OnHttpRequestComplete);

	if (!HttpRequest->ProcessRequest())
	{
		bIsExecuting = false;
		PendingOnComplete.Unbind();
		PendingOnProgress.Unbind();
		ActiveRequest.Reset();
		OnComplete.ExecuteIfBound(TEXT("[LiteLLM] failed to start HTTP request"), false);
		return false;
	}

	return true;
}

// ---------------------------------------------------------------------------
// IClaudeRunner — ExecuteSync (blocking, not ideal — avoid on game thread)
// ---------------------------------------------------------------------------

bool FLiteLLMRunner::ExecuteSync(const FClaudeRequestConfig& Config, FString& OutResponse)
{
	// Simple synchronous implementation using a blocking event.
	// Should only be called from worker threads.
	bool bDone  = false;
	bool bOk    = false;
	FString Result;

	FOnClaudeResponse Callback;
	Callback.BindLambda([&](const FString& Response, bool bSuccess)
	{
		Result  = Response;
		bOk     = bSuccess;
		bDone   = true;
	});

	if (!ExecuteAsync(Config, Callback))
	{
		OutResponse = TEXT("[LiteLLM] ExecuteAsync failed");
		return false;
	}

	// Spin-wait (acceptable on a worker thread; do NOT call from game thread)
	while (!bDone)
	{
		FPlatformProcess::Sleep(0.05f);
	}

	OutResponse = Result;
	return bOk;
}
