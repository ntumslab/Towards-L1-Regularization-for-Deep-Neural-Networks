#ifndef TENSORFLOW_SPARSE_CONV2D_CC_KERNELS_EIGEN_PACKET_MATH_H_
#define TENSORFLOW_SPARSE_CONV2D_CC_KERNELS_EIGEN_PACKET_MATH_H_


#include "third_party/eigen3/Eigen/Core"


namespace Eigen {

namespace internal {

template<typename Packet>
struct mask_traits;

template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
pzero(const Packet& a) { return pxor(a, a); }

template<typename Packet> EIGEN_DEVICE_FUNC inline Packet
pmask_loadu(const typename unpacket_traits<Packet>::type* from, const typename mask_traits<Packet>::type& mask);

template<typename Packet> EIGEN_DEVICE_FUNC inline void
pmask_storeu(typename unpacket_traits<Packet>::type* to, const Packet& from, const typename mask_traits<Packet>::type& mask);


#ifdef EIGEN_VECTORIZE_AVX512

template<>
struct mask_traits<Packet16f> {
  typedef __mmask16 type;
};

template<>
struct mask_traits<Packet8d> {
  typedef __mmask8 type;
};

template<> EIGEN_STRONG_INLINE Packet16f pzero(const Packet16f& /*a*/) { return _mm512_setzero_ps(); }
template<> EIGEN_STRONG_INLINE Packet8d pzero(const Packet8d& /*a*/) { return _mm512_setzero_pd(); }

template<> EIGEN_STRONG_INLINE Packet16f
pmask_loadu<Packet16f>(const float* from, const __mmask16& mask) {
  Packet16f unused;
  return _mm512_mask_loadu_ps(unused, mask, from);
}

template<> EIGEN_STRONG_INLINE Packet8d
pmask_loadu<Packet8d>(const double* from, const __mmask8& mask) {
  Packet8d unused;
  return _mm512_mask_loadu_pd(unused, mask, from);
}

template<> EIGEN_STRONG_INLINE void
pmask_storeu<Packet16f>(float* to, const Packet16f& from, const __mmask16& mask) {
  _mm512_mask_storeu_ps(to, mask, from);
}

template<> EIGEN_STRONG_INLINE void
pmask_storeu<Packet8d>(double* to, const Packet8d& from, const __mmask8& mask) {
  _mm512_mask_storeu_pd(to, mask, from);
}

#endif // EIGEN_VECTORIZE_AVX512


#ifdef EIGEN_VECTORIZE_AVX

template<> EIGEN_STRONG_INLINE Packet8f pzero(const Packet8f& /*a*/) { return _mm256_setzero_ps(); }
template<> EIGEN_STRONG_INLINE Packet4d pzero(const Packet4d& /*a*/) { return _mm256_setzero_pd(); }

#endif // EIGEN_VECTORIZE_AVX


#ifdef EIGEN_VECTORIZE_SSE

template<> EIGEN_STRONG_INLINE Packet4f pzero(const Packet4f& /*a*/) { return _mm_setzero_ps(); }
template<> EIGEN_STRONG_INLINE Packet2d pzero(const Packet2d& /*a*/) { return _mm_setzero_pd(); }

#endif // EIGEN_VECTORIZE_SSE

} // namespace internal

} // namespace Eigen

#endif // TENSORFLOW_SPARSE_CONV2D_CC_KERNELS_EIGEN_PACKET_MATH_H_
