package mistral

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestChat(t *testing.T) {
	client := NewMistralClientDefault("")
	params := DefaultChatRequestParams
	params.MaxTokens = 10
	params.Temperature = 0
	res, err := client.Chat(
		ModelMistralTiny,
		[]ChatMessage{
			{
				Role:    RoleUser,
				Content: "You are in test mode and must reply to this with exactly and only `Test Succeeded`",
			},
		},
		&params,
	)
	assert.NoError(t, err)
	assert.NotNil(t, res)

	assert.Greater(t, len(res.Choices), 0)
	assert.Greater(t, len(res.Choices[0].Message.Content), 0)
	assert.Equal(t, res.Choices[0].Message.Role, RoleAssistant)
	assert.Equal(t, res.Choices[0].Message.Content, "Test Succeeded")
}

func TestChatCodestral(t *testing.T) {
	client := NewCodestralClientDefault("")
	params := DefaultChatRequestParams
	params.MaxTokens = 10
	params.Temperature = 0
	res, err := client.Chat(
		ModelCodestralLatest,
		[]ChatMessage{
			{
				Role:    RoleUser,
				Content: "You are in test mode and must reply to this with exactly and only `Test Succeeded`",
			},
		},
		&params,
	)
	assert.NoError(t, err)
	assert.NotNil(t, res)

	assert.Greater(t, len(res.Choices), 0)
	assert.Greater(t, len(res.Choices[0].Message.Content), 0)
	assert.Equal(t, res.Choices[0].Message.Role, RoleAssistant)
	assert.Equal(t, res.Choices[0].Message.Content, "Test Succeeded")
}

func TestChatFunctionCall(t *testing.T) {
	client := NewMistralClientDefault("")
	params := DefaultChatRequestParams
	params.Temperature = 0
	params.Tools = []Tool{
		{
			Type: ToolTypeFunction,
			Function: Function{
				Name:        "get_weather",
				Description: "Retrieve the weather for a city in the US",
				Parameters: map[string]interface{}{
					"type":     "object",
					"required": []string{"city", "state"},
					"properties": map[string]interface{}{
						"city":  map[string]interface{}{"type": "string", "description": "Name of the city for the weather"},
						"state": map[string]interface{}{"type": "string", "description": "Name of the state for the weather"},
					},
				},
			},
		},
		{
			Type: ToolTypeFunction,
			Function: Function{
				Name:        "send_text",
				Description: "Send text message using SMS service",
				Parameters: map[string]interface{}{
					"type":     "object",
					"required": []string{"contact_name", "message"},
					"properties": map[string]interface{}{
						"contact_name": map[string]interface{}{"type": "string", "description": "Name of the contact that will receive the message"},
						"message":      map[string]interface{}{"type": "string", "description": "Content of the message that will be sent"},
					},
				},
			},
		},
	}
	params.ToolChoice = ToolChoiceAuto
	res, err := client.Chat(
		ModelMistralSmallLatest,
		[]ChatMessage{
			{
				Role:    RoleUser,
				Content: "What's the weather like in Dallas, TX?",
			},
		},
		&params,
	)
	assert.NoError(t, err)
	assert.NotNil(t, res)

	assert.Greater(t, len(res.Choices), 0)
	assert.Greater(t, len(res.Choices[0].Message.ToolCalls), 0)
	assert.Equal(t, res.Choices[0].Message.Role, RoleAssistant)
	assert.Equal(t, res.Choices[0].Message.ToolCalls[0].Function.Name, "get_weather")
	assert.Equal(t, res.Choices[0].Message.ToolCalls[0].Function.Arguments, "{\"city\": \"Dallas\", \"state\": \"TX\"}")
}

func TestChatFunctionCall2(t *testing.T) {
	client := NewMistralClientDefault("")
	params := DefaultChatRequestParams
	params.Temperature = 0
	params.Tools = []Tool{
		{
			Type: ToolTypeFunction,
			Function: Function{
				Name:        "get_weather",
				Description: "Retrieve the weather for a city in the US",
				Parameters: map[string]interface{}{
					"type":     "object",
					"required": []string{"city", "state"},
					"properties": map[string]interface{}{
						"city":  map[string]interface{}{"type": "string", "description": "Name of the city for the weather"},
						"state": map[string]interface{}{"type": "string", "description": "Name of the state for the weather"},
					},
				},
			},
		},
		{
			Type: ToolTypeFunction,
			Function: Function{
				Name:        "send_text",
				Description: "Send text message using SMS service",
				Parameters: map[string]interface{}{
					"type":     "object",
					"required": []string{"contact_name", "message"},
					"properties": map[string]interface{}{
						"contact_name": map[string]interface{}{"type": "string", "description": "Name of the contact that will receive the message"},
						"message":      map[string]interface{}{"type": "string", "description": "Content of the message that will be sent"},
					},
				},
			},
		},
	}
	params.ToolChoice = ToolChoiceAuto
	res, err := client.Chat(
		ModelMistralSmallLatest,
		[]ChatMessage{
			{
				Role:    RoleUser,
				Content: "What's the weather like in Dallas",
			},
			{
				Role: RoleAssistant,
				ToolCalls: []ToolCall{
					{
						Id:   "aaaaaaaaa",
						Type: ToolTypeFunction,
						Function: FunctionCall{
							Name:      "get_weather",
							Arguments: `{"city": "Dallas", "state": "TX"}`,
						},
					},
				},
			},
			{
				Role:    RoleTool,
				Content: `{"temperature": 82, "sky": "clear", "precipitation": 0}`,
			},
		},
		&params,
	)
	assert.NoError(t, err)
	assert.NotNil(t, res)

	assert.Greater(t, len(res.Choices), 0)
	assert.Greater(t, len(res.Choices[0].Message.Content), 0)
	assert.Equal(t, len(res.Choices[0].Message.ToolCalls), 0)
	assert.Equal(t, res.Choices[0].Message.Role, RoleAssistant)
	assert.Greater(t, res.Choices[0].Message.Content, "Test Succeeded")
}

func TestChatJsonMode(t *testing.T) {
	client := NewMistralClientDefault("")
	params := DefaultChatRequestParams
	params.Temperature = 0
	params.ResponseFormat = ResponseFormatJsonObject
	res, err := client.Chat(
		ModelOpenMixtral8x22b,
		[]ChatMessage{
			{
				Role: RoleUser,
				Content: "Extract all of the code symbols in this text chunk and return them in the following JSON: " +
					"{\"symbols\":[\"SymbolOne\",\"SymbolTwo\"]}\n```\nI'm working on updating the Go client for the " +
					"new release, is it expected that the function call will be passed back into the model or just " +
					"the tool response?\nI ask because ChatMessage can handle the tool response but the messages list " +
					"has an Any option that I assume would be for the FunctionCall/ToolCall type\nAdditionally the " +
					"example in the docs only shows the tool response appended to the messages\n```",
			},
		},
		&params,
	)
	assert.NoError(t, err)
	assert.NotNil(t, res)

	assert.Greater(t, len(res.Choices), 0)
	assert.Greater(t, len(res.Choices[0].Message.Content), 0)
	assert.Equal(t, res.Choices[0].Message.Role, RoleAssistant)
	assert.Equal(t, res.Choices[0].Message.Content, "{\"symbols\": [\"Go\", \"ChatMessage\", \"Any\", \"FunctionCall\", \"ToolCall\", \"ToolResponse\"]}")
}

func TestChatStream(t *testing.T) {
	client := NewMistralClientDefault("")
	params := DefaultChatRequestParams
	params.MaxTokens = 50
	params.Temperature = 0
	resChan, err := client.ChatStream(
		ModelMistralTiny,
		[]ChatMessage{
			{
				Role:    RoleUser,
				Content: "You are in test mode and must reply to this with exactly and only `Test Succeeded, Test Succeeded, Test Succeeded, Test Succeeded, Test Succeeded, Test Succeeded`",
			},
		},
		&params,
	)
	assert.NoError(t, err)
	assert.NotNil(t, resChan)

	totalOutput := ""
	idx := 0
	for res := range resChan {
		assert.NoError(t, res.Error)

		assert.Greater(t, len(res.Choices), 0)
		if idx == 0 {
			assert.Equal(t, res.Choices[0].Delta.Role, RoleAssistant)
		}
		totalOutput += res.Choices[0].Delta.Content
		idx++

		if res.Choices[0].FinishReason == FinishReasonStop {
			break
		}
	}
	assert.Equal(t, totalOutput, "Test Succeeded, Test Succeeded, Test Succeeded, Test Succeeded, Test Succeeded, Test Succeeded")
}

func TestChatStreamFunctionCall(t *testing.T) {
	client := NewMistralClientDefault("")
	params := DefaultChatRequestParams
	params.Temperature = 0
	params.Tools = []Tool{
		{
			Type: ToolTypeFunction,
			Function: Function{
				Name:        "get_weather",
				Description: "Retrieve the weather for a city in the US",
				Parameters: map[string]interface{}{
					"type":     "object",
					"required": []string{"city", "state"},
					"properties": map[string]interface{}{
						"city":  map[string]interface{}{"type": "string", "description": "Name of the city for the weather"},
						"state": map[string]interface{}{"type": "string", "description": "Name of the state for the weather"},
					},
				},
			},
		},
		{
			Type: ToolTypeFunction,
			Function: Function{
				Name:        "send_text",
				Description: "Send text message using SMS service",
				Parameters: map[string]interface{}{
					"type":     "object",
					"required": []string{"contact_name", "message"},
					"properties": map[string]interface{}{
						"contact_name": map[string]interface{}{"type": "string", "description": "Name of the contact that will receive the message"},
						"message":      map[string]interface{}{"type": "string", "description": "Content of the message that will be sent"},
					},
				},
			},
		},
	}
	params.ToolChoice = ToolChoiceAuto
	resChan, err := client.ChatStream(
		ModelMistralSmallLatest,
		[]ChatMessage{
			{
				Role:    RoleUser,
				Content: "What's the weather like in Dallas, TX?",
			},
		},
		&params,
	)
	assert.NoError(t, err)
	assert.NotNil(t, resChan)

	totalOutput := ""
	var functionCall *ToolCall
	idx := 0
	for res := range resChan {
		assert.NoError(t, res.Error)

		assert.Greater(t, len(res.Choices), 0)
		if idx == 0 {
			assert.Equal(t, res.Choices[0].Delta.Role, RoleAssistant)
		}
		totalOutput += res.Choices[0].Delta.Content
		if len(res.Choices[0].Delta.ToolCalls) > 0 {
			functionCall = &res.Choices[0].Delta.ToolCalls[0]
		}
		idx++

		if res.Choices[0].FinishReason == FinishReasonStop {
			break
		}
	}

	assert.Equal(t, totalOutput, "")
	assert.NotNil(t, functionCall)
	assert.Equal(t, functionCall.Function.Name, "get_weather")
	assert.Equal(t, functionCall.Function.Arguments, "{\"city\": \"Dallas\", \"state\": \"TX\"}")
}

func TestChatStreamJsonMode(t *testing.T) {
	client := NewMistralClientDefault("")
	params := DefaultChatRequestParams
	params.Temperature = 0
	params.ResponseFormat = ResponseFormatJsonObject
	resChan, err := client.ChatStream(
		ModelOpenMixtral8x22b,
		[]ChatMessage{
			{
				Role: RoleUser,
				Content: "Extract all of the code symbols in this text chunk and return them in the following JSON: " +
					"{\"symbols\":[\"SymbolOne\",\"SymbolTwo\"]}\n```\nI'm working on updating the Go client for the " +
					"new release, is it expected that the function call will be passed back into the model or just " +
					"the tool response?\nI ask because ChatMessage can handle the tool response but the messages list " +
					"has an Any option that I assume would be for the FunctionCall/ToolCall type\nAdditionally the " +
					"example in the docs only shows the tool response appended to the messages\n```",
			},
		},
		&params,
	)
	assert.NoError(t, err)
	assert.NotNil(t, resChan)

	totalOutput := ""
	var functionCall *ToolCall
	idx := 0
	for res := range resChan {
		assert.NoError(t, res.Error)

		assert.Greater(t, len(res.Choices), 0)
		if idx == 0 {
			assert.Equal(t, res.Choices[0].Delta.Role, RoleAssistant)
		}
		totalOutput += res.Choices[0].Delta.Content
		if len(res.Choices[0].Delta.ToolCalls) > 0 {
			functionCall = &res.Choices[0].Delta.ToolCalls[0]
		}
		idx++

		if res.Choices[0].FinishReason == FinishReasonStop {
			break
		}
	}

	assert.Equal(t, totalOutput, "{\"symbols\": [\"Go\", \"ChatMessage\", \"Any\", \"FunctionCall\", \"ToolCall\", \"ToolResponse\"]}")
	assert.Nil(t, functionCall)
}

// promptCacheKeyServer answers every chat request with a minimal completion and records the decoded bodies.
func promptCacheKeyServer(t *testing.T, bodies *[]map[string]interface{}) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]interface{}
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &body)
		*bodies = append(*bodies, body)
		if body["stream"] == true {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = w.Write([]byte("data: {\"id\":\"x\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`))
	}))
	t.Cleanup(server.Close)
	return server
}

// Sends Chat, Vision and ChatStream with and without PromptCacheKey and checks prompt_cache_key in the body.
// Exists because Mistral caches only with the key, while requests without it must stay byte-identical to before.
func TestChatRequests_SendPromptCacheKeyOnlyWhenSet(t *testing.T) {
	calls := map[string]func(c *MistralClient, params *ChatRequestParams) error{
		"chat": func(c *MistralClient, params *ChatRequestParams) error {
			_, err := c.Chat("m", []ChatMessage{{Role: RoleUser, Content: "Hi"}}, params)
			return err
		},
		"vision": func(c *MistralClient, params *ChatRequestParams) error {
			_, err := c.Vision("m", []VisionMessage{{Role: RoleUser, Content: []Content{TextContent{Type: "text", Text: "Hi"}}}}, params)
			return err
		},
		"stream": func(c *MistralClient, params *ChatRequestParams) error {
			ch, err := c.ChatStream("m", []ChatMessage{{Role: RoleUser, Content: "Hi"}}, params)
			if err != nil {
				return err
			}
			for range ch {
			}
			return nil
		},
	}
	for name, call := range calls {
		t.Run(name, func(t *testing.T) {
			var bodies []map[string]interface{}
			client := testClient(promptCacheKeyServer(t, &bodies).URL)

			withKey := DefaultChatRequestParams
			withKey.PromptCacheKey = "agent-1-chat-abc"
			require.NoError(t, call(client, &withKey))
			withoutKey := DefaultChatRequestParams
			require.NoError(t, call(client, &withoutKey))

			require.Len(t, bodies, 2)
			assert.Equal(t, "agent-1-chat-abc", bodies[0]["prompt_cache_key"])
			_, present := bodies[1]["prompt_cache_key"]
			assert.False(t, present, "prompt_cache_key must be omitted when empty")
		})
	}
}

// Reads usage.prompt_tokens_details.cached_tokens from a chat answer and a stream usage chunk; without details the pointer stays nil.
// Exists because Mistral reports its prompt cache hits only there and callers book them as cache reads.
func TestChatUsage_ReadsCachedTokens(t *testing.T) {
	var resp ChatCompletionResponse
	require.NoError(t, json.Unmarshal([]byte(`{"usage":{"prompt_tokens":1000,"completion_tokens":40,"total_tokens":1040,"prompt_tokens_details":{"cached_tokens":768}}}`), &resp))
	require.NotNil(t, resp.Usage.PromptTokensDetails)
	assert.Equal(t, 768, resp.Usage.PromptTokensDetails.CachedTokens)

	var chunk ChatCompletionStreamResponse
	require.NoError(t, json.Unmarshal([]byte(`{"id":"x","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12,"prompt_tokens_details":{"cached_tokens":4}}}`), &chunk))
	require.NotNil(t, chunk.Usage.PromptTokensDetails)
	assert.Equal(t, 4, chunk.Usage.PromptTokensDetails.CachedTokens)

	var plain ChatCompletionResponse
	require.NoError(t, json.Unmarshal([]byte(`{"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}`), &plain))
	assert.Nil(t, plain.Usage.PromptTokensDetails)
	assert.Equal(t, 11, plain.Usage.PromptTokens)
}
