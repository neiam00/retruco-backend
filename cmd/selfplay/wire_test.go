package main

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestServeNegotiatesWithoutDroppingPipelinedBytes(t *testing.T) {
	command := []byte(`[{"op":"reset","id":0,"seed":19,"config":{"players":2,"target":12}}]`)
	input := append([]byte("[{\"op\":\"protocol\",\"protocol\":\"retruco-selfplay-binary-v1\"}]\n"), binary.LittleEndian.AppendUint32(nil, uint32(len(command)))...)
	input = append(input, command...)
	var output bytes.Buffer
	require.NoError(t, serve(bytes.NewReader(input), &output))
	ack, err := output.ReadBytes('\n')
	require.NoError(t, err)
	require.Contains(t, string(ack), binaryProtocol)
	var size uint32
	require.NoError(t, binary.Read(&output, binary.LittleEndian, &size))
	require.Equal(t, int(size), output.Len())
	frame := output.Bytes()
	require.Equal(t, "RTO1", string(frame[:4]))
	metaSize := binary.LittleEndian.Uint32(frame[4:8])
	var rows []map[string]any
	require.NoError(t, json.Unmarshal(frame[8:8+metaSize], &rows))
	observation := rows[0]["observation"].(map[string]any)
	require.NotContains(t, observation, "features")
	require.NotContains(t, observation, "actions")
	require.Equal(t, float64(39), observation["action_count"])
}

func TestServeRejectsTruncatedAndOversizedFrames(t *testing.T) {
	hello := []byte("[{\"op\":\"protocol\",\"protocol\":\"retruco-selfplay-binary-v1\"}]\n")
	for _, test := range []struct {
		name  string
		frame []byte
	}{
		{"partial header", []byte{1, 0}},
		{"missing body", binary.LittleEndian.AppendUint32(nil, 10)},
		{"partial body", append(binary.LittleEndian.AppendUint32(nil, 10), 'x')},
		{"oversized body", binary.LittleEndian.AppendUint32(nil, maxRequestBytes+1)},
		{"empty body", binary.LittleEndian.AppendUint32(nil, 0)},
	} {
		t.Run(test.name, func(t *testing.T) {
			input := append(append([]byte(nil), hello...), test.frame...)
			require.Error(t, serve(bytes.NewReader(input), io.Discard))
		})
	}
}
