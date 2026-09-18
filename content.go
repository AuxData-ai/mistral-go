package mistral

import (
	"bytes"
	"encoding/json"
	"strings"
)

// The Mistral API returns `content` either as a plain string or — for
// reasoning models served through La Plateforme (GLM, Magistral, …) — as an
// array of content chunks:
//
//	"content": [{"type":"thinking","thinking":[{"type":"text","text":"…"}]},
//	            {"type":"text","text":"Hello"}]
//
// Decoding that into the `Content string` field fails with
// "json: cannot unmarshal array into Go struct field … of type string", which
// aborts the whole response (non-streaming) or drops every chunk (streaming).
// The unmarshalers below accept both shapes and flatten an array to the text
// it carries. Non-text chunks (thinking, image_url, reference, document_url)
// are not part of the answer and are dropped.
//
// Marshaling is untouched: requests still send `content` as a string, so the
// public API stays source-compatible.

// chatMessageWire mirrors ChatMessage with a raw content field.
type chatMessageWire struct {
	Role      string          `json:"role"`
	Content   json.RawMessage `json:"content"`
	ToolCalls []ToolCall      `json:"tool_calls"`
}

// UnmarshalJSON accepts `content` as a string or as an array of content chunks.
func (m *ChatMessage) UnmarshalJSON(data []byte) error {
	var wire chatMessageWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	content, err := flattenContent(wire.Content)
	if err != nil {
		return err
	}
	m.Role = wire.Role
	m.Content = content
	m.ToolCalls = wire.ToolCalls
	return nil
}

// UnmarshalJSON accepts `content` as a string or as an array of content chunks.
func (m *DeltaMessage) UnmarshalJSON(data []byte) error {
	var wire chatMessageWire
	if err := json.Unmarshal(data, &wire); err != nil {
		return err
	}
	content, err := flattenContent(wire.Content)
	if err != nil {
		return err
	}
	m.Role = wire.Role
	m.Content = content
	m.ToolCalls = wire.ToolCalls
	return nil
}

// flattenContent turns a raw `content` value into the text it carries.
func flattenContent(raw json.RawMessage) (string, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" {
		return "", nil
	}

	var asString string
	if err := json.Unmarshal(trimmed, &asString); err == nil {
		return asString, nil
	}

	var chunks []json.RawMessage
	if err := json.Unmarshal(trimmed, &chunks); err != nil {
		return "", err
	}
	var text strings.Builder
	for _, chunk := range chunks {
		text.WriteString(textFromChunk(chunk))
	}
	return text.String(), nil
}

// textFromChunk returns the text of a single content chunk. A chunk without a
// `type` counts as text; everything but "text" is dropped.
func textFromChunk(chunk json.RawMessage) string {
	var asString string
	if err := json.Unmarshal(chunk, &asString); err == nil {
		return asString
	}
	var typed struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(chunk, &typed); err != nil {
		return ""
	}
	if typed.Type != "" && typed.Type != "text" {
		return ""
	}
	return typed.Text
}
