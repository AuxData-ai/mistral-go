package mistral

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"mime/multipart"
	"net/textproto"
	"path/filepath"
	"strings"
)

// TimestampGranularity controls the level of timing detail in transcription responses.
// Note: TimestampGranularities and Language are mutually exclusive parameters.
type TimestampGranularity string

const (
	TimestampGranularitySegment TimestampGranularity = "segment"
	TimestampGranularityWord    TimestampGranularity = "word"
)

// TranscriptionRequestParams holds optional parameters for transcription requests.
//
// Constraints:
//   - TimestampGranularities and Language are mutually exclusive.
//   - Diarize is not compatible with TranscribeStream.
type TranscriptionRequestParams struct {
	// Language hint to improve accuracy (e.g. "en", "fr"). Mutually exclusive with TimestampGranularities.
	Language string
	// Diarize enables speaker identification. Not supported in streaming mode.
	Diarize bool
	// Temperature sampling temperature (0–1).
	Temperature float64
	// ContextBias is a list of up to 100 words/phrases to boost domain-specific accuracy.
	ContextBias []string
	// TimestampGranularities requests timing at segment or word level. Mutually exclusive with Language.
	TimestampGranularities []TimestampGranularity
}

// TranscriptionSegment is a timestamped segment of transcribed audio.
type TranscriptionSegment struct {
	ID    int     `json:"id"`
	Seek  int     `json:"seek"`
	Start float64 `json:"start"`
	End   float64 `json:"end"`
	Text  string  `json:"text"`
}

// TranscriptionUsage holds usage statistics for a transcription request.
type TranscriptionUsage struct {
	PromptAudioSeconds float64 `json:"prompt_audio_seconds"`
	PromptTokens       int     `json:"prompt_tokens"`
	CompletionTokens   int     `json:"completion_tokens"`
	TotalTokens        int     `json:"total_tokens"`
}

// TranscriptionResponse is the response from the audio transcription endpoint.
type TranscriptionResponse struct {
	Model    string                 `json:"model"`
	Text     string                 `json:"text"`
	Language string                 `json:"language"`
	Segments []TranscriptionSegment `json:"segments"`
	Usage    TranscriptionUsage     `json:"usage"`
}

// TranscriptionStreamEventType identifies the type of a streaming transcription SSE event.
type TranscriptionStreamEventType string

const (
	TranscriptionEventSessionCreated TranscriptionStreamEventType = "realtime.session.created"
	TranscriptionEventTextDelta      TranscriptionStreamEventType = "transcription.text.delta"
	TranscriptionEventDone           TranscriptionStreamEventType = "transcription.done"
	TranscriptionEventError          TranscriptionStreamEventType = "realtime.error"
)

// TranscriptionStreamEvent is a single SSE event from a streaming transcription response.
type TranscriptionStreamEvent struct {
	Type  TranscriptionStreamEventType `json:"type"`
	Text  string                       `json:"text,omitempty"`
	Error string                       `json:"error,omitempty"`
}

// audioMIMEType returns the MIME content type for a given audio filename.
func audioMIMEType(filename string) string {
	switch strings.ToLower(filepath.Ext(filename)) {
	case ".wav":
		return "audio/wav"
	case ".mp4", ".m4a":
		return "audio/mp4"
	case ".ogg":
		return "audio/ogg"
	case ".flac":
		return "audio/flac"
	case ".webm":
		return "audio/webm"
	default:
		return "audio/mpeg"
	}
}

// buildTranscriptionForm constructs a multipart form body for the transcription endpoint.
// Provide either (file + filename) or fileURL — not both.
func buildTranscriptionForm(model string, file io.Reader, filename, fileURL string, stream bool, params *TranscriptionRequestParams) (*bytes.Buffer, string, error) {
	body := &bytes.Buffer{}
	w := multipart.NewWriter(body)

	if err := w.WriteField("model", model); err != nil {
		return nil, "", err
	}
	if stream {
		if err := w.WriteField("stream", "true"); err != nil {
			return nil, "", err
		}
	}

	switch {
	case fileURL != "":
		if err := w.WriteField("file_url", fileURL); err != nil {
			return nil, "", err
		}
	case file != nil:
		h := make(textproto.MIMEHeader)
		h.Set("Content-Disposition", fmt.Sprintf(`form-data; name="file"; filename="%s"`, filename))
		h.Set("Content-Type", audioMIMEType(filename))
		part, err := w.CreatePart(h)
		if err != nil {
			return nil, "", err
		}
		if _, err = io.Copy(part, file); err != nil {
			return nil, "", err
		}
	}

	if params != nil {
		if params.Language != "" {
			if err := w.WriteField("language", params.Language); err != nil {
				return nil, "", err
			}
		}
		if params.Diarize {
			if err := w.WriteField("diarize", "true"); err != nil {
				return nil, "", err
			}
		}
		if params.Temperature != 0 {
			if err := w.WriteField("temperature", fmt.Sprintf("%g", params.Temperature)); err != nil {
				return nil, "", err
			}
		}
		for _, bias := range params.ContextBias {
			if err := w.WriteField("context_bias", bias); err != nil {
				return nil, "", err
			}
		}
		for _, g := range params.TimestampGranularities {
			if err := w.WriteField("timestamp_granularities[]", string(g)); err != nil {
				return nil, "", err
			}
		}
	}

	if err := w.Close(); err != nil {
		return nil, "", err
	}
	return body, w.FormDataContentType(), nil
}

// Transcribe submits an audio file for offline (batch) transcription.
//
// Supported formats: MP3, WAV, MP4, M4A, OGG, FLAC, WEBM (up to 3 hours).
// Use ModelVoxtralMiniLatest or ModelVoxtralMini2602 as the model.
func (c *MistralClient) Transcribe(model string, file io.Reader, filename string, params *TranscriptionRequestParams) (*TranscriptionResponse, error) {
	body, ct, err := buildTranscriptionForm(model, file, filename, "", false, params)
	if err != nil {
		return nil, err
	}

	response, err := c.requestMultipart(body, ct, "v1/audio/transcriptions", false)
	if err != nil {
		return nil, err
	}

	respData, ok := response.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid response type: %T", response)
	}

	var result TranscriptionResponse
	if err = mapToStruct(respData, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// TranscribeURL submits a remote audio URL for offline (batch) transcription.
//
// Use ModelVoxtralMiniLatest or ModelVoxtralMini2602 as the model.
func (c *MistralClient) TranscribeURL(model string, fileURL string, params *TranscriptionRequestParams) (*TranscriptionResponse, error) {
	body, ct, err := buildTranscriptionForm(model, nil, "", fileURL, false, params)
	if err != nil {
		return nil, err
	}

	response, err := c.requestMultipart(body, ct, "v1/audio/transcriptions", false)
	if err != nil {
		return nil, err
	}

	respData, ok := response.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid response type: %T", response)
	}

	var result TranscriptionResponse
	if err = mapToStruct(respData, &result); err != nil {
		return nil, err
	}
	return &result, nil
}

// TranscribeStream submits an audio file and returns a channel of streaming SSE events.
//
// Use ModelVoxtralMiniRealtime for lowest latency streaming.
// Note: Diarize is not supported in streaming mode.
// The channel is closed when the stream ends or an error occurs.
func (c *MistralClient) TranscribeStream(model string, file io.Reader, filename string, params *TranscriptionRequestParams) (<-chan TranscriptionStreamEvent, error) {
	body, ct, err := buildTranscriptionForm(model, file, filename, "", true, params)
	if err != nil {
		return nil, err
	}

	response, err := c.requestMultipart(body, ct, "v1/audio/transcriptions", true)
	if err != nil {
		return nil, err
	}

	respBody, ok := response.(io.ReadCloser)
	if !ok {
		return nil, fmt.Errorf("invalid response type: %T", response)
	}

	eventChan := make(chan TranscriptionStreamEvent)
	go func() {
		defer close(eventChan)
		defer respBody.Close()

		reader := bufio.NewReader(respBody)
		for {
			line, err := reader.ReadBytes('\n')
			if err == io.EOF {
				break
			} else if err != nil {
				eventChan <- TranscriptionStreamEvent{
					Type:  TranscriptionEventError,
					Error: fmt.Sprintf("error reading stream: %v", err),
				}
				return
			}

			if bytes.Equal(line, []byte("\n")) {
				continue
			}

			if bytes.HasPrefix(line, []byte("data: ")) {
				jsonLine := bytes.TrimSpace(bytes.TrimPrefix(line, []byte("data: ")))
				if bytes.Equal(jsonLine, []byte("[DONE]")) {
					break
				}

				var event TranscriptionStreamEvent
				if err := json.Unmarshal(jsonLine, &event); err != nil {
					eventChan <- TranscriptionStreamEvent{
						Type:  TranscriptionEventError,
						Error: fmt.Sprintf("error decoding event: %v", err),
					}
					continue
				}
				eventChan <- event
			}
		}
	}()

	return eventChan, nil
}
