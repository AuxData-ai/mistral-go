package mistral

import (
	"fmt"
	"net/http"
)

const (
	// ModelVoxtralMiniTTS is Mistral's text-to-speech model.
	ModelVoxtralMiniTTS = "voxtral-mini-tts-2603"

	SpeechFormatPcm  = "pcm"
	SpeechFormatWav  = "wav"
	SpeechFormatMp3  = "mp3"
	SpeechFormatFlac = "flac"
	SpeechFormatOpus = "opus"

	// VoiceTypeAll includes both preset and account-owned custom voices.
	VoiceTypeAll    = "all"
	VoiceTypePreset = "preset"
	VoiceTypeCustom = "custom"
)

// SpeechRequestParams holds the optional parameters of a speech request.
type SpeechRequestParams struct {
	// VoiceId selects a preset or custom voice (see ListVoices).
	VoiceId string
	// ResponseFormat is one of pcm, wav, mp3, flac, opus. Empty defaults to mp3.
	ResponseFormat string
	// RefAudio is a one-off base64 reference clip used to clone a voice.
	RefAudio string
	// PromptCacheKey lets the API reuse a cached prompt across requests.
	PromptCacheKey string
}

// SpeechResponse is the non-streaming response of the speech endpoint.
type SpeechResponse struct {
	// AudioData is the generated audio, base64 encoded in the requested format.
	AudioData string `json:"audio_data"`
}

// Voice describes a preset or custom voice usable for speech generation.
type Voice struct {
	Id          string   `json:"id"`
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Gender      string   `json:"gender"`
	Languages   []string `json:"languages"`
}

// voiceList is the paginated envelope the voices endpoint returns.
type voiceList struct {
	Items []Voice `json:"items"`
	Total int     `json:"total"`
}

// Speech synthesises input with the given TTS model and returns base64 audio.
//
// Use ModelVoxtralMiniTTS as the model. Streaming is not supported here; the
// response is always the non-streaming JSON envelope.
func (c *MistralClient) Speech(model string, input string, params *SpeechRequestParams) (*SpeechResponse, error) {
	if input == "" {
		return nil, fmt.Errorf("input must not be empty")
	}

	requestData := map[string]interface{}{
		"input":  input,
		"stream": false,
	}

	if model != "" {
		requestData["model"] = model
	}

	responseFormat := SpeechFormatMp3

	if params != nil {
		if params.VoiceId != "" {
			requestData["voice_id"] = params.VoiceId
		}
		if params.RefAudio != "" {
			requestData["ref_audio"] = params.RefAudio
		}
		if params.PromptCacheKey != "" {
			requestData["prompt_cache_key"] = params.PromptCacheKey
		}
		if params.ResponseFormat != "" {
			responseFormat = params.ResponseFormat
		}
	}

	requestData["response_format"] = responseFormat

	response, err := c.request(http.MethodPost, requestData, "v1/audio/speech", false, nil)
	if err != nil {
		return nil, err
	}

	respData, ok := response.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid response type: %T", response)
	}

	var result SpeechResponse
	if err = mapToStruct(respData, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

// ListVoices returns the voices available to the API key. voiceType is one of
// VoiceTypeAll, VoiceTypePreset or VoiceTypeCustom; empty defaults to all.
func (c *MistralClient) ListVoices(voiceType string) ([]Voice, error) {
	if voiceType == "" {
		voiceType = VoiceTypeAll
	}

	response, err := c.request(http.MethodGet, nil, "v1/audio/voices", false, map[string]string{"type": voiceType})
	if err != nil {
		return nil, err
	}

	respData, ok := response.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("invalid response type: %T", response)
	}

	var list voiceList
	if err = mapToStruct(respData, &list); err != nil {
		return nil, err
	}

	return list.Items, nil
}
