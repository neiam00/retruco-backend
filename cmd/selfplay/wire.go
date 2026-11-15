package main

import (
	"bufio"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"math"

	"github.com/neiam00/retruco-backend/cpu/learning"
)

const binaryProtocol = "retruco-selfplay-binary-v1"
const maxRequestBytes = 4 << 20
const maxResponseBytes = 64 << 20

// binaryObservation shadows the expensive numeric JSON fields. Metadata stays
// compatible with Observation; its feature ordering remains owned by Go.
type binaryObservation struct {
	*learning.Observation
	Features    any             `json:"features,omitempty"`
	Actions     any             `json:"actions,omitempty"`
	ActionCount int             `json:"action_count"`
	Training    *binaryTraining `json:"training,omitempty"`
}

type binaryTraining struct {
	Version    int                 `json:"version"`
	Memory     learning.MemoryPage `json:"memory"`
	EventCount int                 `json:"event_count"`
}

type binaryResponse struct {
	ID          int                `json:"id"`
	Observation *binaryObservation `json:"observation,omitempty"`
	Error       string             `json:"error,omitempty"`
}

func encodeBinary(results []response) ([]byte, error) {
	metadata := make([]binaryResponse, len(results))
	floats := 0
	for i, result := range results {
		metadata[i] = binaryResponse{ID: result.ID, Error: result.Error}
		if result.Observation != nil {
			o := result.Observation
			if len(o.Actions) > 1024 {
				return nil, fmt.Errorf("too many binary candidates")
			}
			metadata[i].Observation = &binaryObservation{Observation: o, ActionCount: len(o.Actions)}
			floats += learning.StateSize + len(o.Actions)*learning.ActionSize
			if o.Training != nil {
				t := o.Training
				page := t.Memory
				page.Events = nil
				metadata[i].Observation.Training = &binaryTraining{Version: t.Version, Memory: page, EventCount: len(t.Memory.Events)}
				floats += len(t.Memory.Events)*learning.StateSize + learning.CriticSize
			}
		}
	}
	meta, err := json.Marshal(metadata)
	if err != nil {
		return nil, err
	}
	size := 8 + len(meta) + floats*4
	if size > maxResponseBytes {
		return nil, fmt.Errorf("binary response too large")
	}
	data := make([]byte, 0, size+4)
	data = binary.LittleEndian.AppendUint32(data, uint32(size))
	data = append(data, "RTO1"...)
	data = binary.LittleEndian.AppendUint32(data, uint32(len(meta)))
	data = append(data, meta...)
	for _, result := range results {
		if result.Observation == nil {
			continue
		}
		for _, value := range result.Observation.Features {
			data = binary.LittleEndian.AppendUint32(data, math.Float32bits(value))
		}
		for _, action := range result.Observation.Actions {
			for _, value := range action {
				data = binary.LittleEndian.AppendUint32(data, math.Float32bits(value))
			}
		}
		if t := result.Observation.Training; t != nil {
			for _, event := range t.Memory.Events {
				for _, value := range event {
					data = binary.LittleEndian.AppendUint32(data, math.Float32bits(value))
				}
			}
			for _, value := range t.Privileged {
				data = binary.LittleEndian.AppendUint32(data, math.Float32bits(value))
			}
		}
	}
	return data, nil
}

func readLine(reader *bufio.Reader) ([]byte, error) {
	var data []byte
	for {
		part, err := reader.ReadSlice('\n')
		if len(data)+len(part) > maxRequestBytes {
			return nil, fmt.Errorf("JSON request too large")
		}
		data = append(data, part...)
		if err == bufio.ErrBufferFull {
			continue
		}
		if err == io.EOF && len(data) > 0 {
			return data, nil // retain JSON-lines support for a final line without LF
		}
		return data, err
	}
}

func readFrame(reader io.Reader) ([]byte, error) {
	var header [4]byte
	if _, err := io.ReadFull(reader, header[:]); err != nil {
		return nil, err
	}
	size := binary.LittleEndian.Uint32(header[:])
	if size == 0 || size > maxRequestBytes {
		return nil, fmt.Errorf("invalid binary request length")
	}
	data := make([]byte, int(size))
	_, err := io.ReadFull(reader, data)
	if err == io.EOF {
		err = io.ErrUnexpectedEOF
	}
	return data, err
}

func serve(input io.Reader, output io.Writer) error {
	reader := bufio.NewReader(input)
	writer := bufio.NewWriter(output)
	envs := map[int]*learning.Environment{}
	framed := false
	first := true
	for {
		var data []byte
		var err error
		if framed {
			data, err = readFrame(reader)
		} else {
			data, err = readLine(reader)
		}
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}
		var requests []request
		var results []response
		if err := json.Unmarshal(data, &requests); err != nil {
			results = []response{{Error: err.Error()}}
		} else if first && len(requests) == 1 && requests[0].Op == "protocol" && requests[0].Protocol == binaryProtocol {
			// A JSON handshake means an old simulator returns an immediate error,
			// instead of hanging while its new client waits for a binary header.
			ack := map[string]any{"protocol": binaryProtocol, "state_size": learning.StateSize, "action_size": learning.ActionSize}
			if err := json.NewEncoder(writer).Encode(ack); err != nil {
				return err
			}
			if err := writer.Flush(); err != nil {
				return err
			}
			framed, first = true, false
			continue
		} else if framed && (len(requests) == 0 || len(requests) > 64) {
			results = []response{{Error: "binary batch must contain 1 to 64 requests"}}
		} else {
			results = process(requests, envs)
		}
		first = false
		if framed {
			encoded, err := encodeBinary(results)
			if err != nil {
				return err
			}
			_, err = writer.Write(encoded)
			if err != nil {
				return err
			}
		} else if err := json.NewEncoder(writer).Encode(results); err != nil {
			return err
		}
		if err := writer.Flush(); err != nil {
			return err
		}
	}
}
