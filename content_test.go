package mistral

import (
	"encoding/json"
	"testing"
)

// Offline tests — unlike the other tests in this package they need no API key.

func TestChatMessage_UnmarshalStringContent(t *testing.T) {
	var msg ChatMessage
	if err := json.Unmarshal([]byte(`{"role":"assistant","content":"Hallo Welt"}`), &msg); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if msg.Role != "assistant" || msg.Content != "Hallo Welt" {
		t.Fatalf("got %+v", msg)
	}
}

func TestChatMessage_UnmarshalArrayContent(t *testing.T) {
	raw := `{"role":"assistant","content":[
		{"type":"thinking","thinking":[{"type":"text","text":"denke"}]},
		{"type":"text","text":"Hallo "},
		{"type":"text","text":"Welt"}
	],"tool_calls":[{"id":"c1","type":"function","function":{"name":"f","arguments":"{}"}}]}`

	var msg ChatMessage
	if err := json.Unmarshal([]byte(raw), &msg); err != nil {
		t.Fatalf("array content must not fail: %v", err)
	}
	if msg.Content != "Hallo Welt" {
		t.Fatalf("Content = %q, want %q", msg.Content, "Hallo Welt")
	}
	if len(msg.ToolCalls) != 1 || msg.ToolCalls[0].Id != "c1" {
		t.Fatalf("tool calls lost: %+v", msg.ToolCalls)
	}
}

func TestChatMessage_UnmarshalNullAndEmptyContent(t *testing.T) {
	for _, raw := range []string{
		`{"role":"assistant","content":null}`,
		`{"role":"assistant","content":[]}`,
		`{"role":"assistant"}`,
	} {
		var msg ChatMessage
		if err := json.Unmarshal([]byte(raw), &msg); err != nil {
			t.Fatalf("%s: unexpected error: %v", raw, err)
		}
		if msg.Content != "" {
			t.Fatalf("%s: Content = %q, want empty", raw, msg.Content)
		}
	}
}

func TestChatMessage_MarshalStaysAString(t *testing.T) {
	out, err := json.Marshal(ChatMessage{Role: RoleUser, Content: "Frage"})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(out) != `{"role":"user","content":"Frage"}` {
		t.Fatalf("request encoding changed: %s", out)
	}
}

func TestDeltaMessage_UnmarshalArrayContent(t *testing.T) {
	var delta DeltaMessage
	raw := `{"content":[{"type":"text","text":"Teil"}]}`
	if err := json.Unmarshal([]byte(raw), &delta); err != nil {
		t.Fatalf("array content must not fail: %v", err)
	}
	if delta.Content != "Teil" {
		t.Fatalf("Content = %q, want %q", delta.Content, "Teil")
	}
}

func TestChatCompletionResponse_UnmarshalArrayContent(t *testing.T) {
	raw := `{"id":"x","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":[{"type":"text","text":"Hallo Welt"}]},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}`

	var resp ChatCompletionResponse
	if err := json.Unmarshal([]byte(raw), &resp); err != nil {
		t.Fatalf("array content must not fail: %v", err)
	}
	if len(resp.Choices) != 1 || resp.Choices[0].Message.Content != "Hallo Welt" {
		t.Fatalf("got %+v", resp.Choices)
	}
	if resp.Choices[0].FinishReason != FinishReasonStop {
		t.Fatalf("finish reason = %q", resp.Choices[0].FinishReason)
	}
}

func TestFlattenContent_PlainStringArray(t *testing.T) {
	got, err := flattenContent(json.RawMessage(`["a","b"]`))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "ab" {
		t.Fatalf("got %q, want %q", got, "ab")
	}
}

func TestFlattenContent_RejectsUnsupportedShape(t *testing.T) {
	if _, err := flattenContent(json.RawMessage(`{"unexpected":true}`)); err == nil {
		t.Fatal("an object content must still be reported as an error")
	}
}
