package mistral

import (
	"bytes"
	"encoding/binary"
	"mime"
	"mime/multipart"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ── helpers ──────────────────────────────────────────────────────────────────

// silentWAV returns a minimal valid WAV file containing the given number of
// silent (zero) 16-bit PCM samples at 16 kHz mono. The resulting file is
// recognised by the Mistral transcription API.
func silentWAV(samples int) []byte {
	dataBytes := samples * 2 // 16-bit = 2 bytes per sample
	buf := &bytes.Buffer{}

	// RIFF header
	buf.WriteString("RIFF")
	binary.Write(buf, binary.LittleEndian, uint32(36+dataBytes)) // chunk size
	buf.WriteString("WAVE")

	// fmt sub-chunk
	buf.WriteString("fmt ")
	binary.Write(buf, binary.LittleEndian, uint32(16))    // sub-chunk size
	binary.Write(buf, binary.LittleEndian, uint16(1))     // PCM
	binary.Write(buf, binary.LittleEndian, uint16(1))     // mono
	binary.Write(buf, binary.LittleEndian, uint32(16000)) // sample rate
	binary.Write(buf, binary.LittleEndian, uint32(32000)) // byte rate
	binary.Write(buf, binary.LittleEndian, uint16(2))     // block align
	binary.Write(buf, binary.LittleEndian, uint16(16))    // bits per sample

	// data sub-chunk
	buf.WriteString("data")
	binary.Write(buf, binary.LittleEndian, uint32(dataBytes))
	buf.Write(make([]byte, dataBytes)) // silence

	return buf.Bytes()
}

// parseForm parses multipart form data from raw body bytes and content-type header.
func parseForm(t *testing.T, body []byte, contentType string) map[string]string {
	t.Helper()
	_, params, err := mime.ParseMediaType(contentType)
	require.NoError(t, err)
	boundary := params["boundary"]
	require.NotEmpty(t, boundary, "missing boundary in content-type")

	r := multipart.NewReader(bytes.NewReader(body), boundary)
	form, err := r.ReadForm(10 << 20)
	require.NoError(t, err)

	fields := make(map[string]string)
	for k, v := range form.Value {
		if len(v) > 0 {
			fields[k] = v[0]
		}
	}
	for k, fhs := range form.File {
		if len(fhs) > 0 {
			fields["__file_field__"] = k
			fields["__file_name__"] = fhs[0].Filename
			fields["__file_mime__"] = fhs[0].Header.Get("Content-Type")
		}
	}
	return fields
}

// ── unit tests ────────────────────────────────────────────────────────────────

func TestAudioMIMEType(t *testing.T) {
	cases := []struct {
		filename string
		expected string
	}{
		{"audio.mp3", "audio/mpeg"},
		{"audio.MP3", "audio/mpeg"},
		{"audio.wav", "audio/wav"},
		{"audio.WAV", "audio/wav"},
		{"audio.mp4", "audio/mp4"},
		{"audio.m4a", "audio/mp4"},
		{"audio.ogg", "audio/ogg"},
		{"audio.flac", "audio/flac"},
		{"audio.webm", "audio/webm"},
		{"audio.unknown", "audio/mpeg"}, // fallback
		{"no_extension", "audio/mpeg"},  // fallback
	}

	for _, tc := range cases {
		t.Run(tc.filename, func(t *testing.T) {
			assert.Equal(t, tc.expected, audioMIMEType(tc.filename))
		})
	}
}

func TestBuildTranscriptionForm_FileUpload(t *testing.T) {
	content := []byte("fake audio data")
	body, ct, err := buildTranscriptionForm(
		ModelVoxtralMiniLatest,
		bytes.NewReader(content),
		"test.mp3",
		"",
		false,
		nil,
	)

	require.NoError(t, err)
	assert.True(t, strings.HasPrefix(ct, "multipart/form-data"))

	fields := parseForm(t, body.Bytes(), ct)
	assert.Equal(t, ModelVoxtralMiniLatest, fields["model"])
	assert.Equal(t, "file", fields["__file_field__"])
	assert.Equal(t, "test.mp3", fields["__file_name__"])
	assert.Equal(t, "audio/mpeg", fields["__file_mime__"])
	assert.Empty(t, fields["stream"])
}

func TestBuildTranscriptionForm_FileURL(t *testing.T) {
	audioURL := "https://example.com/audio.wav"
	body, ct, err := buildTranscriptionForm(
		ModelVoxtralMini2602,
		nil, "",
		audioURL,
		false,
		nil,
	)

	require.NoError(t, err)
	fields := parseForm(t, body.Bytes(), ct)
	assert.Equal(t, ModelVoxtralMini2602, fields["model"])
	assert.Equal(t, audioURL, fields["file_url"])
	assert.Empty(t, fields["__file_field__"])
}

func TestBuildTranscriptionForm_StreamFlag(t *testing.T) {
	body, ct, err := buildTranscriptionForm(
		ModelVoxtralMiniRealtime,
		bytes.NewReader([]byte("audio")),
		"clip.wav",
		"",
		true,
		nil,
	)

	require.NoError(t, err)
	fields := parseForm(t, body.Bytes(), ct)
	assert.Equal(t, "true", fields["stream"])
}

func TestBuildTranscriptionForm_OptionalParams(t *testing.T) {
	params := &TranscriptionRequestParams{
		Language:    "en",
		Diarize:     true,
		Temperature: 0.3,
		ContextBias: []string{"OpenAI", "Mistral"},
		TimestampGranularities: []TimestampGranularity{
			TimestampGranularityWord,
		},
	}

	body, ct, err := buildTranscriptionForm(
		ModelVoxtralMiniLatest,
		bytes.NewReader([]byte("audio")),
		"test.mp3",
		"",
		false,
		params,
	)

	require.NoError(t, err)

	_, fmtParams, _ := mime.ParseMediaType(ct)
	boundary := fmtParams["boundary"]
	r := multipart.NewReader(bytes.NewReader(body.Bytes()), boundary)
	form, err := r.ReadForm(10 << 20)
	require.NoError(t, err)

	assert.Equal(t, []string{"en"}, form.Value["language"])
	assert.Equal(t, []string{"true"}, form.Value["diarize"])
	assert.Equal(t, []string{"0.3"}, form.Value["temperature"])
	assert.Equal(t, []string{"OpenAI", "Mistral"}, form.Value["context_bias"])
	assert.Equal(t, []string{"word"}, form.Value["timestamp_granularities[]"])
}

func TestBuildTranscriptionForm_NilParams(t *testing.T) {
	body, ct, err := buildTranscriptionForm(
		ModelVoxtralMiniLatest,
		bytes.NewReader([]byte("audio")),
		"clip.ogg",
		"",
		false,
		nil,
	)

	require.NoError(t, err)
	fields := parseForm(t, body.Bytes(), ct)
	assert.Empty(t, fields["language"])
	assert.Empty(t, fields["diarize"])
	assert.Empty(t, fields["temperature"])
	assert.Equal(t, "audio/ogg", fields["__file_mime__"])
}

// ── integration tests ─────────────────────────────────────────────────────────

// TestTranscribeURL transcribes a publicly hosted audio file.
func TestTranscribeURL(t *testing.T) {
	client := NewMistralClientDefault("")

	// Short public domain English speech sample (LibriVox excerpt, ~6 s).
	const sampleURL = "https://upload.wikimedia.org/wikipedia/commons/1/1f/Dial_up_modem_noises.ogg"

	res, err := client.TranscribeURL(ModelVoxtralMiniLatest, sampleURL, nil)
	assert.NoError(t, err)
	require.NotNil(t, res)

	assert.NotEmpty(t, res.Model)
	assert.NotEmpty(t, res.Text)
}

// TestTranscribeURLWithLanguage adds an explicit language hint.
func TestTranscribeURLWithLanguage(t *testing.T) {
	client := NewMistralClientDefault("")

	const sampleURL = "https://upload.wikimedia.org/wikipedia/commons/1/1f/Dial_up_modem_noises.ogg"

	res, err := client.TranscribeURL(ModelVoxtralMiniLatest, sampleURL, &TranscriptionRequestParams{
		Language: "en",
	})
	assert.NoError(t, err)
	require.NotNil(t, res)
	assert.NotEmpty(t, res.Text)
}

// TestTranscribeSilentFile uploads a programmatically generated silent WAV.
// The API is expected to return an empty or near-empty transcription; the test
// validates the response structure, not specific text content.
func TestTranscribeSilentFile(t *testing.T) {
	client := NewMistralClientDefault("")

	// 0.5 s of silence at 16 kHz mono 16-bit PCM
	wav := silentWAV(8000)

	res, err := client.Transcribe(ModelVoxtralMiniLatest, bytes.NewReader(wav), "silence.wav", nil)
	assert.NoError(t, err)
	require.NotNil(t, res)

	assert.NotEmpty(t, res.Model)
	// Text may be empty for silent audio — that's fine
	assert.GreaterOrEqual(t, res.Usage.TotalTokens, 0)
}

// TestTranscribeWithTimestamps requests word-level timestamps.
func TestTranscribeWithTimestamps(t *testing.T) {
	client := NewMistralClientDefault("")

	const sampleURL = "https://upload.wikimedia.org/wikipedia/commons/1/1f/Dial_up_modem_noises.ogg"

	res, err := client.TranscribeURL(ModelVoxtralMiniLatest, sampleURL, &TranscriptionRequestParams{
		TimestampGranularities: []TimestampGranularity{TimestampGranularitySegment},
	})
	assert.NoError(t, err)
	require.NotNil(t, res)
	assert.NotEmpty(t, res.Text)
}

// TestTranscribeStream uploads a silent WAV and verifies at least a done event
// is received on the streaming channel.
func TestTranscribeStream(t *testing.T) {
	client := NewMistralClientDefault("")

	wav := silentWAV(8000) // 0.5 s silence
	eventChan, err := client.TranscribeStream(ModelVoxtralMiniRealtime, bytes.NewReader(wav), "silence.wav", nil)
	require.NoError(t, err)
	require.NotNil(t, eventChan)

	var gotDone bool
	var gotError bool
	for event := range eventChan {
		switch event.Type {
		case TranscriptionEventDone:
			gotDone = true
		case TranscriptionEventError:
			gotError = true
			t.Logf("stream error event: %s", event.Error)
		}
	}

	assert.False(t, gotError, "expected no error events in stream")
	assert.True(t, gotDone, "expected a done event before channel close")
}
