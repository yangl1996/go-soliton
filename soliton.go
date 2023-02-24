// Package soliton implements Soliton and Robust Soliton distributions as
// proposed in paper "LT codes" by Michael Luby.
package soliton

import (
	"math"
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
	var sum []float64
	tot := 0.0
	var i uint64
	for i = 1; i <= k; i++ {
		sum = append(sum, tot)
		tot += (rho(k, i) + tau(c, delta, k, i))
	}
	for i = 1; i < k; i++ {
		sum[i] /= tot
	}
	sum = append(sum, 1.0)
	return &Soliton{src, k, sum[1:]}
}

// tau implements the function tau for the robust Soliton distribution
func tau(c, delta float64, k, i uint64) float64 {
	r := ripple(c, delta, k)
	th := uint64(math.Round(float64(k) / r))
	if i < th {                              // 1 to k/R-1
		return r / float64(i * k)
	} else if i == th { // k/R
		return r * (math.Log(r) - math.Log(delta)) / float64(k)
	} else { // k/R+1 to k
		return 0
	}
}

// ripple calculates the expected ripple size of a robust soliton distribution.
func ripple(c, delta float64, k uint64) float64 {
	kf := float64(k)
	res := c * math.Log(kf/delta) * math.Sqrt(kf)
	return res
}

// rho implements the rho(i) function in soliton distribution.
func rho(k, i uint64) float64 {
	if i == 1 {
		return 1.0 / float64(k)
	} else {
		return 1.0 / float64(i * (i-1))
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
	last := 0.0
	var i uint64
	for i = 1; i < k; i++ { // we only do 1 to k-1 (incl.) because we only need k-1 splits
		p := rho(k, i)
		last += p
		s = append(s, last)
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

// PMF returns the probability mass function as a slice of float64, where the i-th
// element is the probability of i+1.
func (s *Soliton) PMF() []float64 {
	last := 0.0
	res := []float64{}
	for _, p := range s.splits {
		res = append(res, p-last)
		last = p
	}
	return res
}
