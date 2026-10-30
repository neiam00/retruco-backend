package main

import "testing"

func TestPasswordDutyConfiguration(t *testing.T) {
	for _, value := range []string{"0", "101", "bad", "-1"} {
		t.Run(value, func(t *testing.T) {
			t.Setenv("PASSWORD_WORK_DUTY_PERCENT", value)
			if _, err := passwordDuty(); err == nil {
				t.Fatal("invalid duty accepted")
			}
		})
	}
	t.Setenv("PASSWORD_WORK_DUTY_PERCENT", "")
	if got, err := passwordDuty(); err != nil || got != 20 {
		t.Fatalf("default: %d %v", got, err)
	}
}
