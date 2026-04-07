// Copyright Natali Caggiano. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "IClaudeRunner.h"
#include "Interfaces/IHttpRequest.h"
#include "Interfaces/IHttpResponse.h"

/**
 * LiteLLM configuration loaded from Config/litellm-config.json
 */
struct FLiteLLMConfig
{
	/** Base URL of the LiteLLM proxy, e.g. https://litellm.example.com */
	FString BaseUrl;

	/** Bearer auth token */
	FString AuthToken;

	/** Model name to pass to LiteLLM, e.g. gpt-5.4 */
	FString Model;

	bool IsValid() const { return !BaseUrl.IsEmpty() && !AuthToken.IsEmpty() && !Model.IsEmpty(); }
};

/**
 * IClaudeRunner implementation that calls a LiteLLM (OpenAI-compatible) proxy
 * instead of launching the local Claude Code CLI.
 *
 * Config is read from <ProjectDir>/Config/litellm-config.json:
 * {
 *   "base_url": "https://...",
 *   "auth_token": "sk-...",
 *   "model": "gpt-..."
 * }
 */
class FLiteLLMRunner : public IClaudeRunner
{
public:
	explicit FLiteLLMRunner(const FLiteLLMConfig& InConfig);
	virtual ~FLiteLLMRunner();

	// IClaudeRunner interface
	virtual bool ExecuteAsync(
		const FClaudeRequestConfig& Config,
		FOnClaudeResponse OnComplete,
		FOnClaudeProgress OnProgress = FOnClaudeProgress()
	) override;

	virtual bool ExecuteSync(const FClaudeRequestConfig& Config, FString& OutResponse) override;
	virtual void Cancel() override;
	virtual bool IsExecuting() const override { return bIsExecuting; }
	virtual bool IsAvailable() const override { return LiteLLMConfig.IsValid(); }

	/** Try to load config from the given path. Returns false if file is absent or malformed. */
	static bool TryLoadConfig(const FString& ConfigFilePath, FLiteLLMConfig& OutConfig);

private:
	/** Build the JSON body for /v1/chat/completions */
	FString BuildRequestBody(const FClaudeRequestConfig& Config) const;

	/** Parse the assistant content out of a chat-completions response body */
	FString ParseResponseBody(const FString& ResponseBody) const;

	/** Called on HTTP request completion */
	void OnHttpRequestComplete(FHttpRequestPtr Request, FHttpResponsePtr Response, bool bWasSuccessful);

	FLiteLLMConfig LiteLLMConfig;

	TAtomic<bool> bIsExecuting{false};

	// Stored callbacks for the in-flight request
	FOnClaudeResponse PendingOnComplete;
	FOnClaudeProgress PendingOnProgress;

	// Keep a reference so we can cancel
	FHttpRequestPtr ActiveRequest;
};
