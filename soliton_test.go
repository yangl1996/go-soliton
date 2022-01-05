package soliton

import (
	"math"
	"math/rand"
	"testing"
	"strconv"
)

// constRNG implements rand.Source and always returns one value.
type constRNG struct {
	val int64
}

func (c constRNG) Seed(seed int64) {
}

func (c constRNG) Int63() int64 {
	return c.val
}

var rng = rand.New(rand.NewSource(0))

func BenchmarkSample(b *testing.B) {
	ks := []int{10, 50, 100, 200, 1000, 10000, 100000}
	for _, k := range ks {
		name := strconv.Itoa(k)
		dist := NewSoliton(rng, uint64(k))
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				dist.Uint64()
			}
		})
	}
}

// TestUint64 tests the sampling of soliton distribution.
func TestUint64(t *testing.T) {
	r1 := rand.New(constRNG{0})
	s1 := NewSoliton(r1, 3)
	if s1.Uint64() != 1 {
		t.Error("wrong sampled value, should be 1")
	}

	var norm float64
	norm = 1<<63
	r2 := rand.New(constRNG{int64(0.35 * norm)})
	s2 := NewSoliton(r2, 3)
	if s2.Uint64() != 2 {
		t.Error("wrong sampled value, should be 2")
	}

	r3 := rand.New(constRNG{int64(0.9 * norm)})
	s3 := NewSoliton(r3, 3)
	if s3.Uint64() != 3 {
		t.Error("wrong sampled value, should be 3")
	}
}

// TestNewSoliton tests the creation of a soliton distribution.
func TestNewSoliton(t *testing.T) {
	s1 := NewSoliton(rng, 1)
	if len(s1.splits) != 1 || s1.splits[0] != 1.0 {
		t.Error("wrong soliton distribution for k=1")
	}

	s2 := NewSoliton(rng, 2)
	if len(s2.splits) != 2 {
		t.Error("wrong soliton distribution for k=2")
	}
	if s2.splits[0] != 0.5 || s2.splits[1] != 1.0 {
		t.Error("wrong soliton distribution for k=2")
	}

	s3 := NewSoliton(rng, 3)
	if len(s3.splits) != 3 {
		t.Error("wrong soliton distribution for k=3")
	}
	if s3.splits[0] != (1.0/3.0) || s3.splits[1] != (1.0/3.0+0.5) || s3.splits[2] != 1.0 {
		t.Error("wrong soliton distribution for k=3")
	}
}

// TestNewRobustSoliton tests the creation of a robust soliton distribution.
func TestNewRobustSoliton(t *testing.T) {
	s1 := NewRobustSoliton(rng, 1, 1.2, 0.001)
	if len(s1.splits) != 1 || s1.splits[0] != 1.0 {
		t.Error("wrong soliton distribution for k=1")
	}

	s2 := NewRobustSoliton(rng, 3, 0.12, 0.001)
	// R = 1.6640922493490253214333366552672665558709784486254181035332
	// Threshold = 1.80 = 2
	// Tau = 0.5546974164496751071444455517557555186236594828751393678444
	//       4.1142101844753662286537231656606822751670098629800196368954
	//       0
	// Rho = 1/3
	//       1/2
	//       1/6
	// Tau + Rho = 0.8880307497830084404777788850890888519569928162084727011777333333
	//             4.6142101844753662286537231656606822751670098629800196368954
	//             1/6
	// Sum = 5.6689076009250413357981687174164377937906693458551590047397999999
	if len(s2.splits) != 3 {
		t.Error("wrong soliton distribution for k=3")
	}
	t.Log(s2.splits[0])
	t.Log(s2.splits[1])
	t.Log(s2.splits[2])
	e1 := math.Abs(s2.splits[0]-0.1566493603878993030143281532615261342211803355562430962538559755) < 0.000001
	e2 := math.Abs(s2.splits[1]-0.9705998618429641852703276471816809044716254319807433471959590362) < 0.000001
	e3 := math.Abs(s2.splits[2]-1.0) < 0.000001

	if !(e1 && e2 && e3) {
		t.Error("wrong soliton distribution for k=3")
	}
}

// TestSolitonUint64 tests drawing uint64 values from soliton distribution.
func TestSolitonUint64(t *testing.T) {
	s := NewSoliton(rng, 1)
	r := s.Uint64()
	if r != 1 {
		t.Error("drawing from k=1 soliton distribution is not 1")
	}

	// test the sanity check that k should be larger than k
	// we want to test if the function panicked. we want it to panic
	defer func() {
		if r := recover(); r == nil {
			t.Error("uint64 did not panic when returning a value larger than k")
		}
	}()
	s.k = 0
	s.Uint64()
}

// TestSolitonEqual tests the comparator of two Soliton distributions.
func TestSolitonEqual(t *testing.T) {
	s1 := NewSoliton(rng, 4)
	s2 := NewSoliton(rng, 4)
	if s1.Equals(s2) != true {
		t.Error("comparator returns false when two distributions equal")
	}

	s3 := NewSoliton(rng, 5)
	if s1.Equals(s3) != false {
		t.Error("comparator returns true when two distributions differ")
	}
	s4 := NewSoliton(rng, 5)
	s4.k = 4 // we want to trigger the slice length check
	if s1.Equals(s4) != false {
		t.Error("comparator returns true when two distributions differ")
	}
	s5 := NewRobustSoliton(rng, 4, 0.01, 0.02)
	if s1.Equals(s5) != false {
		t.Error("comparator returns true when two distributions differ")
	}

}
