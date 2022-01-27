// Package soliton implements Soliton distribution as proposed in LT code.
package soliton

import (
	"math"
	"math/big"
	"math/rand"
	"sort"
)

// Soliton generates Soliton distributed variates.
type Soliton struct {
	r      *rand.Rand
	k      uint64
	splits []float64 // the entire range of [0, 1) is cut into k pieces with k-1 splits
}

// NewRobustSoliton returns a Robust Soliton variate generator that uses
// src as the source of randomness. See
// https://en.wikipedia.org/wiki/Soliton_distribution#Robust_distribution
// for definitions of the parameters k, c, and delta.
func NewRobustSoliton(src *rand.Rand, k uint64, c, delta float64) *Soliton {
	var sum []*big.Float
	tot := new(big.Float)
	var i uint64
	for i = 1; i <= k; i++ {
		x := new(big.Float).Copy(tot)
		sum = append(sum, x)
		tot.Add(tot, rho(k, i))
		tot.Add(tot, tau(c, delta, k, i))
	}
	var s []float64
	for i = 1; i < k; i++ {
		val, _ := new(big.Float).Quo(sum[i], tot).Float64()
		s = append(s, val)
	}
	s = append(s, 1.0)
	return &Soliton{src, k, s}
}

// tau implements the function tau for the robust Soliton distribution
func tau(c, delta float64, k, i uint64) *big.Float {
	r := ripple(c, delta, k)
	rf := new(big.Float).SetFloat64(r)
	th := uint64(math.Round(float64(k) / r)) // k/R
	if i < th {                              // 1 to k/R-1
		ik := new(big.Float).SetUint64(i * k)
		return new(big.Float).Quo(rf, ik)
	} else if i == th { // k/R
		log := math.Log(r) - math.Log(delta)
		logf := new(big.Float).SetFloat64(log)
		r1 := new(big.Float).Mul(rf, logf)
		return new(big.Float).Quo(r1, new(big.Float).SetUint64(k))
	} else { // k/R+1 to k
		return new(big.Float).SetUint64(0)
	}
}

// ripple calculates the expected ripple size of a robust soliton distribution.
func ripple(c, delta float64, k uint64) float64 {
	kf := float64(k)
	res := c * math.Log(kf/delta) * math.Sqrt(kf)
	return res
}

// rho implements the rho(i) function in soliton distribution.
func rho(k, i uint64) *big.Float {
	if i == 1 {
		one := new(big.Float).SetFloat64(1.0)
		div := new(big.Float).SetUint64(k)
		return new(big.Float).Quo(one, div)
	} else {
		one := new(big.Float).SetFloat64(1.0)
		t1 := new(big.Float).SetUint64(i)
		t2 := new(big.Float).SetUint64(i - 1)
		div := new(big.Float).Mul(t1, t2)
		return new(big.Float).Quo(one, div)
	}
}

/*
NewSoliton returns a Soliton variate generator that uses
src as the source of randomness. The distribution has a single
parameter, k. The PDF is given by
	p(1) = 1/k
	P(i) = 1/(i*(i-1)).

See https://en.wikipedia.org/wiki/Soliton_distribution for a
more detailed description of Soliton and related distributions.
*/
func NewSoliton(src *rand.Rand, k uint64) *Soliton {
	var s []float64
	last := new(big.Float).SetUint64(0)
	var i uint64
	for i = 1; i < k; i++ { // we only do 1 to k-1 (incl.) because we only need k-1 splits
		p := rho(k, i)
		last = last.Add(last, p)
		rounded, _ := last.Float64()
		s = append(s, rounded)
	}
	s = append(s, 1.0)
	return &Soliton{src, k, s}
}

// Uint64 returns a value drawn from the Soliton or Robust Soliton distribution
// described by the Soliton object.
func (s *Soliton) Uint64() uint64 {
	r := s.r.Float64()
	idx := sort.SearchFloat64s(s.splits, r)
	if uint64(idx) >= s.k {
		panic("r should never be larger than the last item in s")
	}
	return uint64(idx + 1)
}

// Equals compares the two soliton distributions by comparing the partition.
func (s *Soliton) Equals(s2 *Soliton) bool {
	if s.k != s2.k {
		return false
	}
	if len(s.splits) != len(s2.splits) {
		return false
	}
	for i := 0; i < len(s.splits); i++ {
		if s.splits[i] != s2.splits[i] {
			return false
		}
	}
	return true
}

// Mean calculates the mean of a Soliton distribution.
func (s *Soliton) Mean() float64 {
	res := 0.0
	lastCdf := 0.0
	for i := range s.splits {
		res += (s.splits[i]-lastCdf) * float64(i+1)
		lastCdf = s.splits[i]
	}
	res += (1.0-lastCdf) * float64(s.k)
	return res
}
