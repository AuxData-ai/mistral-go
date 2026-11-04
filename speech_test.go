package mistral

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// testClient points a client at the given test server with a single attempt,
// so failing status codes surface immediately instead of being retried.
func testClient(url string) *MistralClient {
	return NewMistralClient("test-key", url, 1, 5*time.Second)
}

func TestSpeech_SendsExpectedRequestAndParsesAudio(t *testing.T) {
	var gotPath string
	var gotBody map[string]interface{}
	var gotAuth string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotAuth = r.Header.Get("Authorization")
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"audio_data":"QUJD"}`))
	}))
	defer server.Close()

	resp, err := testClient(server.URL).Speech("voxtral-mini-tts-2603", "Guten Tag", &SpeechRequestParams{
		VoiceId: "voice-1",
	})

	require.NoError(t, err)
	assert.Equal(t, "QUJD", resp.AudioData)
	assert.Equal(t, "/v1/audio/speech", gotPath)
	assert.Equal(t, "Bearer test-key", gotAuth)
	assert.Equal(t, "voxtral-mini-tts-2603", gotBody["model"])
	assert.Equal(t, "Guten Tag", gotBody["input"])
	assert.Equal(t, "voice-1", gotBody["voice_id"])
	assert.Equal(t, "mp3", gotBody["response_format"])
	assert.Equal(t, false, gotBody["stream"])
}

func TestSpeech_HonoursExplicitResponseFormat(t *testing.T) {
	var gotBody map[string]interface{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(raw, &gotBody)
		_, _ = w.Write([]byte(`{"audio_data":"QQ=="}`))
	}))
	defer server.Close()

	_, err := testClient(server.URL).Speech("m", "text", &SpeechRequestParams{ResponseFormat: SpeechFormatWav})

	require.NoError(t, err)
	assert.Equal(t, "wav", gotBody["response_format"])
}

func TestSpeech_RejectsEmptyInput(t *testing.T) {
	_, err := testClient("http://127.0.0.1:1").Speech("m", "", nil)
	require.Error(t, err)
}

func TestSpeech_PropagatesApiError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte(`{"message":"content moderation"}`))
	}))
	defer server.Close()

	_, err := testClient(server.URL).Speech("m", "text", nil)

	require.Error(t, err)
	assert.Contains(t, err.Error(), "403")
	assert.Contains(t, err.Error(), "content moderation")
}

func TestListVoices_SendsTypeQueryAndParsesItems(t *testing.T) {
	var gotPath string
	var gotType string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotPath = r.URL.Path
		gotType = r.URL.Query().Get("type")
		_, _ = w.Write([]byte(`{"items":[
			{"id":"v1","name":"Amelie","description":"warm","gender":"female","languages":["fr"]},
			{"id":"v2","name":"Oliver","languages":["en"]}
		],"total":2}`))
	}))
	defer server.Close()

	voices, err := testClient(server.URL).ListVoices("")

	require.NoError(t, err)
	assert.Equal(t, "/v1/audio/voices", gotPath)
	assert.Equal(t, "all", gotType)
	require.Len(t, voices, 2)
	assert.Equal(t, "v1", voices[0].Id)
	assert.Equal(t, "Amelie", voices[0].Name)
	assert.Equal(t, []string{"fr"}, voices[0].Languages)
	assert.Equal(t, "Oliver", voices[1].Name)
}

// voxtral-mini-tts-2603 ("speech v1") only accepts voices of kind "ivc", so a
// caller needs kind and type to tell usable voices from newer presets.
func TestListVoices_ParsesKindAndType(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"items":[
			{"id":"v1","name":"Heidi","kind":"pvc","type":"preset"},
			{"id":"v2","name":"Auxy","kind":"ivc","type":"custom"}
		],"total":2}`))
	}))
	defer server.Close()

	voices, err := testClient(server.URL).ListVoices("")

	require.NoError(t, err)
	require.Len(t, voices, 2)
	assert.Equal(t, "pvc", voices[0].Kind)
	assert.Equal(t, "preset", voices[0].Type)
	assert.Equal(t, "ivc", voices[1].Kind)
	assert.Equal(t, "custom", voices[1].Type)
}

// The voices endpoint is paginated (default page size 10). A custom voice can
// sit behind dozens of presets, so ListVoices must follow every page.
func TestListVoices_FollowsAllPages(t *testing.T) {
	const total = 25
	const pageSize = 10
	var requests int
	var gotLimit string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		gotLimit = r.URL.Query().Get("limit")
		offset, _ := strconv.Atoi(r.URL.Query().Get("offset"))

		items := []string{}
		for i := offset; i < total && i < offset+pageSize; i++ {
			items = append(items, fmt.Sprintf(`{"id":"v%d","name":"Voice %d"}`, i, i))
		}
		_, _ = fmt.Fprintf(w, `{"items":[%s],"total":%d}`, strings.Join(items, ","), total)
	}))
	defer server.Close()

	voices, err := testClient(server.URL).ListVoices("")

	require.NoError(t, err)
	require.Len(t, voices, total)
	assert.Equal(t, "v0", voices[0].Id)
	assert.Equal(t, "v24", voices[total-1].Id)
	assert.Equal(t, 3, requests)
	assert.Equal(t, "100", gotLimit)
}

// A page without items ends the loop even if total claims more, so a
// misbehaving server cannot make ListVoices spin forever.
func TestListVoices_StopsOnEmptyPage(t *testing.T) {
	var requests int

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests++
		if r.URL.Query().Get("offset") == "0" {
			_, _ = w.Write([]byte(`{"items":[{"id":"v0"}],"total":50}`))
			return
		}
		_, _ = w.Write([]byte(`{"items":[],"total":50}`))
	}))
	defer server.Close()

	voices, err := testClient(server.URL).ListVoices("")

	require.NoError(t, err)
	require.Len(t, voices, 1)
	assert.Equal(t, 2, requests)
}

func TestListVoices_PassesExplicitType(t *testing.T) {
	var gotType string

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotType = r.URL.Query().Get("type")
		_, _ = w.Write([]byte(`{"items":[]}`))
	}))
	defer server.Close()

	voices, err := testClient(server.URL).ListVoices("preset")

	require.NoError(t, err)
	assert.Equal(t, "preset", gotType)
	assert.Empty(t, voices)
}
