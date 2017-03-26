//
//  AccelerateArrayFloat.swift
//  MAS1
//
//  Created by Eric Cole on 3/25/17.
//  Copyright © 2017 Balance Software. All rights reserved.
//

import Accelerate

extension BufferCollection where BufferElement : IsFloat {
	public func buffer() -> UnsafePointer<Float> { return rawPointer().assumingMemoryBound(to:Float.self) }
}

extension MutableBufferCollection where BufferElement : IsFloat {
	public mutating func mutableBuffer() -> UnsafeMutablePointer<Float> { return mutableRawPointer().assumingMemoryBound(to:Float.self) }
}

//	MARK: -

extension MutableBufferCollection where BufferElement : IsFloat {
	/// Transpose matrix into self so that each `self[i][j] = matrix[j][i]`.
	///
	/// In row major order, the number of rows is the major dimension and the number of columns is the minor dimension.
	///
	/// - Parameters:
	///   - matrix: Matrix to transpose.
	///   - major: Major dimension of matrix, minor dimension of self on return.
	///   - minor: Minor dimension of matrix, major dimension of self on return.
	/// - Returns: self containing transposed matrix.
	@discardableResult
	public mutating func vTranspose<C : BufferCollection>(of matrix:C, major:Int, minor:Int) -> Self where C.BufferElement : IsFloat {
		vDSP_mtrans(matrix.buffer(), 1, mutableBuffer(), 1, vDSP_Length(minor), vDSP_Length(major)) ; return self
	}
	
	/// Multiply matrix a by matrix b; each element of the result is the dot product of the corresponding row of a and column of b.
	///
	/// The minor dimension of the result equals the minor dimension of matrix b.  The major dimension of the result equals the major dimension of matrix a, or less if there was not enough space.  If the minor dimension of a does not equal the major dimension of b, the lesser dimension will be used.
	///
	/// In row major order, the number of rows is the major dimension and the number of columns is the minor dimension.
	///
	/// - Parameters:
	///   - a: The first matrix. The resulting matrix will have the same major dimension as a if there is enough space in this buffer.
	///   - b: The second matrix. The resulting matrix will have the same minor dimension as b.
	/// - Returns: self containing the product of the matrix multiplication of a times b, the major and minor dimensions of the product.
	@discardableResult
	public mutating func vProduct<C : BufferCollection, D : BufferCollection>(of a:(matrix:C, minor:Int), times b:(matrix:D, minor:Int)) -> (matrix:Self, major:Int, minor:Int) where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		let minor = b.minor
		let other = Swift.min(a.minor, b.matrix.vCount / b.minor)
		let major = Swift.min(a.matrix.vCount / a.minor, vCount / minor)
		vDSP_mmul(a.matrix.buffer(), 1, b.matrix.buffer(), 1, mutableBuffer(), 1, vDSP_Length(major), vDSP_Length(minor), vDSP_Length(other))
		return (self, major, minor)
	}
	
	public mutating func vPlusProduct<C : BufferCollection, D : BufferCollection>(of a:(matrix:C, minor:Int), times b:(matrix:D, minor:Int), times factor:Float = 1, scalar:Float = 1, size:(major:Int, minor:Int) = (0, 0), matrixMinor:Int = 0, options:Int = 0) -> (matrix:Self, major:Int, minor:Int) where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		let minor = Swift.min(b.minor, size.minor > 0 ? size.minor : Int.max)
		let other = Swift.min(a.minor, b.matrix.vCount / b.minor)
		let major = Swift.min(a.matrix.vCount / a.minor, vCount / (matrixMinor > 0 ? matrixMinor : minor), size.major > 0 ? size.major : Int.max)
		
		cblas_sgemm(
			CblasRowMajor, CblasNoTrans, CblasNoTrans,
			Int32(major), Int32(minor), Int32(other),
			factor, a.matrix.buffer(), Int32(a.minor), b.matrix.buffer(), Int32(b.minor),
			scalar, mutableBuffer(), Int32(matrixMinor > 0 ? matrixMinor : minor)
		)
		
		return (self, major, minor)
	}
	
	/// Assign part of another matrix to part of this matrix.
	///
	/// In row major order, the number of rows is the major dimension and the number of columns is the minor dimension.
	///
	/// - Parameters:
	///   - matrix: The source matrix.
	///   - size: The maximum number of rows and columns to assign. This will be clipped to available dimensions.
	///   - minor: The minor dimension of this matrix.
	///   - from: The position in the source matrix of the first element to copy.
	///   - at: The position in the target matrix of the first element to assign.
	/// - Returns: self containing assigned elements from matrix, the actual size assigned.
	@discardableResult
	public mutating func vAssign<C : BufferCollection>(_ matrix:(matrix:C, minor:Int), size:(major:Int, minor:Int), minor:Int, from:(major:Int, minor:Int) = (0, 0), at:(major:Int, minor:Int) = (0, 0)) -> (matrix:Self, major:Int, minor:Int) where C.BufferElement : IsFloat {
		let move_minor = Swift.min(size.minor, matrix.minor - from.minor, minor - at.minor)
		let move_major = Swift.min(size.major, matrix.matrix.vCount / matrix.minor - from.major, vCount / minor - at.major)
		if move_minor > 0 && move_major > 0 {
			let source = matrix.matrix.buffer().advanced(by:from.major * matrix.minor + from.minor)
			let target = mutableBuffer().advanced(by:at.major * minor + at.minor)
			vDSP_mmov(source, target, vDSP_Length(move_minor), vDSP_Length(move_major), vDSP_Length(matrix.minor), vDSP_Length(minor))
		}
		return (self, move_major, move_minor)
	}
	
	/// Swap elements at major positions a and b.  When the regions overlap then rotate elements from major position a to b.
	///
	/// - Parameters:
	///   - a: The index of the source group of elements.
	///   - b: The index of the target group of elements.
	///   - minor: The minor dimension of the matrix. Elements per column in row major order. 1 for vectors.
	///   - count: The number of minor length groups of elements to move.
	/// - Returns: self with elements swapped or rotated from position a to b.
	public mutating func vSwapMajor(a:UInt, b:UInt, minor:Int, count:Int = 1) -> Self {
		guard minor > 0 && count > 0 && a != b else { return self }
		
		let major = UInt(vCount / minor)
		let upper = Swift.max(a, b)
		let lower = upper ^ a ^ b
		let minor = UInt(minor)
		let count = UInt(count)
		
		guard upper < major else { return self }
		
		let buffer = mutableBuffer()
		let length = upper - lower
		
		if count > length {
			let pivot = a < b ? a + count : b
			
			vDSP_vrvrs(buffer.advanced(by:Int(a * minor)), 1, vDSP_Length(Swift.min(major - a, count) * minor))
			vDSP_vrvrs(buffer.advanced(by:Int(pivot * minor)), 1, vDSP_Length(Swift.min(major - pivot, length) * minor))
			vDSP_vrvrs(buffer.advanced(by:Int(lower * minor)), 1, vDSP_Length(Swift.min(major - lower, length + count) * minor))
		} else {
			vDSP_vswap(buffer.advanced(by:Int(a * minor)), 1, buffer.advanced(by:Int(b * minor)), 1, vDSP_Length(Swift.min(major - upper, count) * minor))
		}
		
		return self
	}
	
	@discardableResult
	public mutating func vConvolution<C : BufferCollection, D : BufferCollection>(of a:(matrix:C, minor:Int), with3x3 b:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		let minor = a.minor
		let major = (Swift.min(vCount, a.matrix.vCount) / (minor * 2)) * 2
		if minor >= 4 && major >= 3 && b.vCount >= 9 {
			vDSP_f3x3(a.matrix.buffer(), vDSP_Length(major), vDSP_Length(minor), b.buffer(), mutableBuffer())
		}
		return self
	}
	
	@discardableResult
	public mutating func vConvolution<C : BufferCollection, D : BufferCollection>(of a:(matrix:C, minor:Int), with5x5 b:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		let minor = a.minor
		let major = (Swift.min(vCount, a.matrix.vCount) / (minor * 2)) * 2
		if minor >= 6 && major >= 5 && b.vCount >= 25 {
			vDSP_f5x5(a.matrix.buffer(), vDSP_Length(major), vDSP_Length(minor), b.buffer(), mutableBuffer())
		}
		return self
	}
	
	@discardableResult
	public mutating func vConvolution<C : BufferCollection, D : BufferCollection>(of a:(matrix:C, minor:Int), with b:(matrix:D, minor:Int)) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		let minor = a.minor
		let major = Swift.min(vCount, a.matrix.vCount) / minor
		vDSP_imgfir(a.matrix.buffer(), vDSP_Length(major), vDSP_Length(minor), b.matrix.buffer(), mutableBuffer(), vDSP_Length(b.matrix.vCount / b.minor), vDSP_Length(b.minor))
		return self
	}
	
	//	MARK: -
	
	/// - Returns: self[i] = -a[i]
	@discardableResult
	public mutating func vNegative<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		vDSP_vneg(a.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount))) ; return self
	}
	
	/// - Returns: self[i] = |a[i]|
	@discardableResult
	public mutating func vMagnitude<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		vDSP_vabs(a.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount))) ; return self
		//vvfabsf(mutableBuffer(), a.buffer(), &length)
	}
	
	/// - Returns: self[i] = -|a[i]|
	@discardableResult
	public mutating func vNegativeMagnitude<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		vDSP_vnabs(a.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount))) ; return self
	}
	
	/// - Returns: self[i] = 1/a[i]
	@discardableResult
	public mutating func vReciprocal<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvrecf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = a[i] + b[i]
	@discardableResult
	public mutating func vSum<C : BufferCollection, D : BufferCollection>(of a:C, plus b:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		vDSP_vadd(a.buffer(), 1, b.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount))) ; return self
	}
	
	/// - Returns: self[i] = a[i] + b
	@discardableResult
	public mutating func vSum<C : BufferCollection>(of a:C, plus b:Float) -> Self where C.BufferElement : IsFloat { var b = b
		vDSP_vsadd(a.buffer(), 1, &b, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount))) ; return self
	}
	
	/// - Returns: self[i] = a[i] * b[i] + c[i]
	@discardableResult
	public mutating func vSum<C : BufferCollection, D : BufferCollection, E : BufferCollection>(of a:C, times b:D, plus c:E) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat, E.BufferElement : IsFloat {
		vDSP_vma(a.buffer(), 1, b.buffer(), 1, c.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount, c.vCount))) ; return self
	}
	
	/// - Returns: self[i] = a[i] * b + c[i]
	@discardableResult
	public mutating func vSum<C : BufferCollection, D : BufferCollection>(of a:C, times b:Float, plus c:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat { var b = b
		vDSP_vsma(a.buffer(), 1, &b, c.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, c.vCount))) ; return self
	}
	
	/// - Returns: self[i] = a[i] * b + c
	@discardableResult
	public mutating func vSum<C : BufferCollection>(of a:C, times b:Float, plus c:Float) -> Self where C.BufferElement : IsFloat { var b = b, c = c
		vDSP_vsmsa(a.buffer(), 1, &b, &c, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount))) ; return self
	}
	
	/// - Returns: self[i] = a[i] * b[i] + c
	@discardableResult
	public mutating func vSum<C : BufferCollection, D : BufferCollection>(of a:C, times b:D, plus c:Float) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat { var c = c
		vDSP_vmsa(a.buffer(), 1, b.buffer(), 1, &c, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount))) ; return self
	}
	
	/// - Returns: self[i] = a[i] * b[i] + c[i] * d[i]
	@discardableResult
	public mutating func vSum<C : BufferCollection, D : BufferCollection, E : BufferCollection, F : BufferCollection>(of a:C, times b:D, plus c:E, times d:F) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat, E.BufferElement : IsFloat, F.BufferElement : IsFloat {
		vDSP_vmma(a.buffer(), 1, b.buffer(), 1, c.buffer(), 1, d.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount, c.vCount, d.vCount))) ; return self
	}
	
	/// - Returns: self[i] = a[i] - b[i]
	@discardableResult
	public mutating func vDifference<C : BufferCollection, D : BufferCollection>(of a:C, minus b:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		vDSP_vsub(b.buffer(), 1, a.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount))) ; return self
	}
	
	/// - Returns: self[i] = a[i] * b[i] - c[i]
	@discardableResult
	public mutating func vDifference<C : BufferCollection, D : BufferCollection, E : BufferCollection>(of a:C, times b:D, minus c:E) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat, E.BufferElement : IsFloat {
		vDSP_vmsb(a.buffer(), 1, b.buffer(), 1, c.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount, c.vCount))) ; return self
	}
	
	/// - Returns: self[i] = a[i] * b[i] - c[i] * d[i]
	@discardableResult
	public mutating func vDifference<C : BufferCollection, D : BufferCollection, E : BufferCollection, F : BufferCollection>(of a:C, times b:D, minus c:E, times d:F) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat, E.BufferElement : IsFloat, F.BufferElement : IsFloat {
		vDSP_vmmsb(a.buffer(), 1, b.buffer(), 1, c.buffer(), 1, d.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount, c.vCount, d.vCount))) ; return self
	}
	
	/// - Returns: self[i] = a[i] * b - c[i]
	@discardableResult
	public mutating func vDifference<C : BufferCollection, D : BufferCollection>(of a:C, times b:Float, minus c:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat { var b = b
		vDSP_vsmsb(a.buffer(), 1, &b, c.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, c.vCount))) ; return self
	}
	
	/// - Returns: self[i] = a[i] * b[i]
	@discardableResult
	public mutating func vProduct<C : BufferCollection, D : BufferCollection>(of a:C, times b:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		vDSP_vmul(a.buffer(), 1, b.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount))) ; return self
	}
	
	/// - Returns: self[i] = a[i] * b
	@discardableResult
	public mutating func vProduct<C : BufferCollection>(of a:C, times b:Float) -> Self where C.BufferElement : IsFloat { var b = b
		vDSP_vsmul(a.buffer(), 1, &b, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount))) ; return self
	}
	
	/// - Returns: self[i] = (a[i] + b[i]) * c[i]
	@discardableResult
	public mutating func vProduct<C : BufferCollection, D : BufferCollection, E : BufferCollection>(of a:C, plus b:D, times c:E) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat, E.BufferElement : IsFloat {
		vDSP_vam(a.buffer(), 1, b.buffer(), 1, c.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount, c.vCount))) ; return self
	}
	
	/// - Returns: self[i] = (a[i] - b[i]) * c[i]
	@discardableResult
	public mutating func vProduct<C : BufferCollection, D : BufferCollection, E : BufferCollection>(of a:C, minus b:D, times c:E) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat, E.BufferElement : IsFloat {
		vDSP_vsbm(a.buffer(), 1, b.buffer(), 1, c.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount, c.vCount))) ; return self
	}
	
	/// - Returns: self[i] = (a[i] + b[i]) * c
	@discardableResult
	public mutating func vProduct<C : BufferCollection, D : BufferCollection>(of a:C, plus b:D, times c:Float) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat { var c = c
		vDSP_vasm(a.buffer(), 1, b.buffer(), 1, &c, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount))) ; return self
	}
	
	/// - Returns: self[i] = (a[i] - b[i]) * c
	@discardableResult
	public mutating func vProduct<C : BufferCollection, D : BufferCollection>(of a:C, minus b:D, times c:Float) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat { var c = c
		vDSP_vsbsm(a.buffer(), 1, b.buffer(), 1, &c, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount))) ; return self
	}
	
	/// - Returns: self[i] = (a[i] + b[i]) * (c[i] + d[i])
	@discardableResult
	public mutating func vProduct<C : BufferCollection, D : BufferCollection, E : BufferCollection, F : BufferCollection>(of a:C, plus b:D, timesQuantity c:E, plus d:F) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat, E.BufferElement : IsFloat, F.BufferElement : IsFloat {
		vDSP_vaam(a.buffer(), 1, b.buffer(), 1, c.buffer(), 1, d.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount, c.vCount, d.vCount))) ; return self
	}
	
	/// - Returns: self[i] = (a[i] - b[i]) * (c[i] - d[i])
	@discardableResult
	public mutating func vProduct<C : BufferCollection, D : BufferCollection, E : BufferCollection, F : BufferCollection>(of a:C, minus b:D, timesQuantity c:E, minus d:F) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat, E.BufferElement : IsFloat, F.BufferElement : IsFloat {
		vDSP_vsbsbm(a.buffer(), 1, b.buffer(), 1, c.buffer(), 1, d.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount, c.vCount, d.vCount))) ; return self
	}
	
	/// - Returns: self[i] = (a[i] + b[i]) * (c[i] - d[i])
	@discardableResult
	public mutating func vProduct<C : BufferCollection, D : BufferCollection, E : BufferCollection, F : BufferCollection>(of a:C, plus b:D, timesQuantity c:E, minus d:F) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat, E.BufferElement : IsFloat, F.BufferElement : IsFloat {
		vDSP_vasbm(a.buffer(), 1, b.buffer(), 1, d.buffer(), 1, c.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount, c.vCount, d.vCount))) ; return self
	}
	
	/// - Returns: self[i] = a[i] ÷ b[i]
	@discardableResult
	public mutating func vQuotient<C : BufferCollection, D : BufferCollection>(of a:C, over b:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		vDSP_vdiv(b.buffer(), 1, a.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount))) ; return self
		//var length = Int32(Swift.min(vCount, a.vCount, b.vCount)) ; vvdivf(mutableBuffer(), a.buffer(), b.buffer(), &length)
	}
	
	/// - Returns: self[i] = a[i] ÷ b
	@discardableResult
	public mutating func vQuotient<C : BufferCollection>(of a:C, over b:Float) -> Self where C.BufferElement : IsFloat { var b = b
		vDSP_vsdiv(a.buffer(), 1, &b, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount))) ; return self
	}
	
	/// - Returns: self[i] = a ÷ b[i]
	@discardableResult
	public mutating func vQuotient<C : BufferCollection>(of a:Float, over b:C) -> Self where C.BufferElement : IsFloat { var a = a
		vDSP_svdiv(&a, b.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, b.vCount))) ; return self
	}
	
	/// - Returns: self[i] = a[i] - b[i] * k ; sign of numerator, magnitude less than denominator
	@discardableResult
	public mutating func vRemainder<C : BufferCollection, D : BufferCollection>(of a:C, modulus b:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount, b.vCount)) ; vvfmodf(mutableBuffer(), a.buffer(), b.buffer(), &length) ; return self
	}
	
	///	- Returns: self[i] = a[i] - b[i] * vnintf(a[i] / b[i]) ; sign based on rounding towards even quotient e.g. 3/2 has remainder -1
	@discardableResult
	public mutating func vRemainder<C : BufferCollection, D : BufferCollection>(of a:C, over b:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount, b.vCount)) ; vvremainderf(mutableBuffer(), a.buffer(), b.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = (self[i] * b + a[i]) / (b + 1)
	@discardableResult
	public mutating func vLinearAverage<C : BufferCollection>(of a:C, with b:Float) -> Self where C.BufferElement : IsFloat { var b = b
		vDSP_vavlin(a.buffer(), 1, &b, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount))) ; return self
	}
	
	/// - Returns: self[i] = max(a[i], b[i])
	@discardableResult
	public mutating func vMaximum<C : BufferCollection, D : BufferCollection>(of a:C, or b:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		vDSP_vmax(a.buffer(), 1, b.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount))) ; return self
	}
	
	/// - Returns: self[i] = max(|a[i]|, |b[i]|)
	@discardableResult
	public mutating func vMaximumMagnitude<C : BufferCollection, D : BufferCollection>(of a:C, or b:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		vDSP_vmaxmg(a.buffer(), 1, b.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount))) ; return self
	}
	
	/// - Returns: self[i] = in(a[i], b[i])
	@discardableResult
	public mutating func vMinimum<C : BufferCollection, D : BufferCollection>(of a:C, or b:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		vDSP_vmin(a.buffer(), 1, b.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount))) ; return self
	}
	
	/// - Returns: self[i] = min(|a[i]|, |b[i]|)
	@discardableResult
	public mutating func vMinimumMagnitude<C : BufferCollection, D : BufferCollection>(of a:C, or b:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		vDSP_vminmg(a.buffer(), 1, b.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount))) ; return self
	}
	
	//	MARK: -
	
	/// - Returns: self[i] = ∑ a[i ..< i+windowLength]
	@discardableResult
	public mutating func vSlidingSum<C : BufferCollection>(of a:C, windowLength:Int) -> Self where C.BufferElement : IsFloat {
		let length = Swift.min(vCount, a.vCount - windowLength + 1)
		if windowLength > 0 && windowLength <= length { vDSP_vswsum(a.buffer(), 1, mutableBuffer(), 1, vDSP_Length(length), vDSP_Length(windowLength)) } ; return self
	}
	
	/// - Returns: self[i] = max(a[i ..< i+windowLength])
	@discardableResult
	public mutating func vSlidingMaximum<C : BufferCollection>(of a:C, windowLength:Int) -> Self where C.BufferElement : IsFloat {
		let length = Swift.min(vCount, a.vCount)
		if windowLength > 0 && windowLength <= length { vDSP_vswmax(a.buffer(), 1, mutableBuffer(), 1, vDSP_Length(length - windowLength + 1), vDSP_Length(windowLength)) } ; return self
	}
	
	/// - Returns: self[i, i+1] = √(a[i]² + a[i+1]²), atan2(a[i+1], a[i])
	@discardableResult
	public mutating func vPolarPoints<C : BufferCollection>(fromCartesian a:C) -> Self where C.BufferElement : IsFloat {
		vDSP_polar(a.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount) / 2)) ; return self
	}
	
	/// - Returns: self[i, i+1] = a[i]*cos(a[i+1]), a[i]*sin(a[i+1])
	@discardableResult
	public mutating func vCartesianPoints<C : BufferCollection>(fromPolar a:C) -> Self where C.BufferElement : IsFloat {
		vDSP_rect(a.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount) / 2)) ; return self
	}
	
	/// - Returns: self[i] = √a[i]
	@discardableResult
	public mutating func vSquareRoot<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvsqrtf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = 1 ÷ √a[i]
	@discardableResult
	public mutating func vOneOverSquareRoot<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvrsqrtf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = ³√a[i]
	@discardableResult
	public mutating func vCubeRoot<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvcbrtf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Parameter a: exponents
	/// - Returns: self[i] = e^a[i]
	@discardableResult
	public mutating func vExpoential<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvexpf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Parameter a: exponents
	/// - Returns: self[i] = (e^a[i]) - 1
	@discardableResult
	public mutating func vExpoential<C : BufferCollection>(minusOne a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvexpm1f(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Parameter a: exponents
	/// - Returns: self[i] = 2^a[i]
	@discardableResult
	public mutating func vBaseTwoExpoential<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvexp2f(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Parameter a: values
	/// - Returns: self[i] = ln(a[i])
	@discardableResult
	public mutating func vLogarithm<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvlogf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Parameter a: values
	/// - Returns: self[i] = ln(a[i] + 1)
	@discardableResult
	public mutating func vLogarithm<C : BufferCollection>(onePlus a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvlog1pf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Parameter a: values
	/// - Returns: self[i] = log10(a[i])
	@discardableResult
	public mutating func vLogarithmBaseTen<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvlog10f(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Parameter a: values
	/// - Returns: self[i] = log2(a[i])
	@discardableResult
	public mutating func vLogarithmBaseTwo<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvlog2f(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Parameter a: values
	/// - Returns: self[i] = floor(log2(a[i]))
	@discardableResult
	public mutating func vBinaryLogarithm<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvlogbf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Parameter a: base
	/// - Parameter b: exponent
	/// - Returns: self[i] = a[i]^b
	@discardableResult
	public mutating func vRaise<C : BufferCollection>(_ a:C, toPower b:Float) -> Self where C.BufferElement : IsFloat { var b = b
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvpowsf(mutableBuffer(), &b, a.buffer(), &length) ; return self
	}
	
	/// - Parameter a: base
	/// - Parameter b: exponent
	/// - Returns: self[i] = a[i]^b[i]
	@discardableResult
	public mutating func vRaise<C : BufferCollection, D : BufferCollection>(_ a:C, toPower b:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount, b.vCount)) ; vvpowf(mutableBuffer(), b.buffer(), a.buffer(), &length) ; return self
	}
	
	/// - Parameter a: base
	/// - Returns: self[i] = a[i]²
	@discardableResult
	public mutating func vSquare<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		vDSP_vsq(a.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount))) ; return self
	}
	
	/// - Parameter a: base
	/// - Returns: self[i] = a[i] * |a[i]|
	@discardableResult
	public mutating func vSignPreservingSquare<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		vDSP_vssq(a.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount))) ; return self
	}
	
	/// - Returns: self[i] = √(a[i]² + b[i]²)
	@discardableResult
	public mutating func vDistance<C : BufferCollection, D : BufferCollection>(of a:C, and b:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		vDSP_vdist(a.buffer(), 1, b.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount))) ; return self
	}
	
	/// - Returns: self[i] = sine(a[i])
	@discardableResult
	public mutating func sin<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvsinf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = cosine(a[i])
	@discardableResult
	public mutating func cos<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvcosf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = tangent(a[i])
	@discardableResult
	public mutating func tan<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvtanf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i, i+n] = sine(a[i]), cosine(a[i])
	@discardableResult
	public mutating func sincos<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount / 2, a.vCount)) ; let p = mutableBuffer() ; vvsincosf(p, p.advanced(by:Int(length)), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = arcsine(a[i])
	@discardableResult
	public mutating func asin<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvasinf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = arccosine(a[i])
	@discardableResult
	public mutating func acos<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvacosf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = arctangent(a[i])
	@discardableResult
	public mutating func atan<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvatanf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = hyperbolic sine(a[i])
	@discardableResult
	public mutating func sinh<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvsinhf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = hyperbolic cosine(a[i])
	@discardableResult
	public mutating func cosh<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvcoshf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = hyperbolic tangent(a[i])
	@discardableResult
	public mutating func tanh<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvtanhf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = inverse hyperbolic sine(a[i])
	@discardableResult
	public mutating func asinh<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvasinhf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = inverse hyperbolic cosine(a[i])
	@discardableResult
	public mutating func acosh<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvacoshf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = inverse hyperbolic tangent(a[i])
	@discardableResult
	public mutating func atanh<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvatanhf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = sine(π•a[i])
	@discardableResult
	public mutating func sin<C : BufferCollection>(piTimes a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvsinpif(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = cosine(π•a[i])
	@discardableResult
	public mutating func cos<C : BufferCollection>(piTimes a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvcospif(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = tangent(π•a[i])
	@discardableResult
	public mutating func tan<C : BufferCollection>(piTimes a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvtanpif(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = trunc(a[i]) ; integer round towards zero
	@discardableResult
	public mutating func vTruncate<C : BufferCollection>(_ a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvintf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = round(a[i]) ; integer round towards nearest
	@discardableResult
	public mutating func vRound<C : BufferCollection>(_ a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvnintf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = ceil(a[i]) ; integer round towards infinity
	@discardableResult
	public mutating func vCeiling<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvceilf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = floor(a[i]) ; integer round towards negative infinity
	@discardableResult
	public mutating func vFloor<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount)) ; vvfloorf(mutableBuffer(), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = a[i] > lower ? a[i] < upper ? a[i] : upper : lower
	@discardableResult
	public mutating func vClip<C : BufferCollection>(of a:C, outside range:ClosedRange<Float>) -> Self where C.BufferElement : IsFloat { var lower = range.lowerBound, upper = range.upperBound
		vDSP_vclip(a.buffer(), 1, &lower, &upper, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount))) ; return self
	}
	
	/// - Returns: self[i], count, count = a[i] > lower ? a[i] < upper ? a[i] : upper : lower
	public mutating func vClipCount<C : BufferCollection>(of a:C, outside range:ClosedRange<Float>) -> (clipped:Self, below:vDSP_Length, above:vDSP_Length) where C.BufferElement : IsFloat { var lower = range.lowerBound, upper = range.upperBound
		var below = vDSP_Length(0), above = vDSP_Length(0); vDSP_vclipc(a.buffer(), 1, &lower, &upper, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount)), &below, &above) ; return (self, below, above)
	}
	
	/// - Returns: self[i] = range.contains(a[i]) ? a[i] < 0 ? lower : upper : a[i]
	@discardableResult
	public mutating func vClip<C : BufferCollection>(of a:C, inside range:ClosedRange<Float>) -> Self where C.BufferElement : IsFloat { var lower = range.lowerBound, upper = range.upperBound
		vDSP_viclip(a.buffer(), 1, &lower, &upper, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount))) ; return self
	}
	
	/// - Returns: self[i] = a[i] < lower ? lower : a[i]
	@discardableResult
	public mutating func vClip<C : BufferCollection>(of a:C, below lower:Float) -> Self where C.BufferElement : IsFloat { var lower = lower
		vDSP_vthr(a.buffer(), 1, &lower, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount))) ; return self
	}
	
	/// - Returns: self[i] = a[i] < lower ? 0 : a[i]
	@discardableResult
	public mutating func vClip<C : BufferCollection>(of a:C, zeroingBelow lower:Float) -> Self where C.BufferElement : IsFloat { var lower = lower
		vDSP_vthres(a.buffer(), 1, &lower, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount))) ; return self
	}
	
	/// - Returns: self[i] = a[i] < lower ? -value : value
	@discardableResult
	public mutating func vThreshold<C : BufferCollection>(of a:C, below lower:Float, value:Float = 1) -> Self where C.BufferElement : IsFloat { var lower = lower, value = value
		vDSP_vthrsc(a.buffer(), 1, &lower, &value, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount))) ; return self
		//vDSP_vlim(a.buffer(), 1, &lower, &value, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount))) ; return self
	}
	
	/// - Returns: self[i] = a[i] < lower[i] || a[i] > upper[i] ? a[i] : 0
	@discardableResult
	public mutating func vEnvelope<C : BufferCollection, D : BufferCollection, E : BufferCollection>(of a:C, zeroBetween lower:D, and upper:E) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat, E.BufferElement : IsFloat {
		vDSP_venvlp(upper.buffer(), 1, lower.buffer(), 1, a.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, lower.vCount, upper.vCount))) ; return self
	}
	
	/// - Returns: self[i] = a[i] + (b[i] - a[i]) * f
	@discardableResult
	public mutating func vInterpolate<C : BufferCollection, D : BufferCollection>(between a:C, and b:D, at factor:Float) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat { var factor = factor
		vDSP_vintb(a.buffer(), 1, b.buffer(), 1, &factor, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount))) ; return self
	}
	
	/// - Returns: self[i] = ∑ a[n] * b[i]ⁿ
	@discardableResult
	public mutating func vEvaluatePolynomial<C : BufferCollection, D : BufferCollection>(coefficients a:C, variables b:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		vDSP_vpoly(a.buffer(), 1, b.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, b.vCount)), vDSP_Length(a.vCount - 1)) ; return self
	}
	
	/// - Returns: self[i] = a[⌊b[i]⌋] + (a[⌊b[i]⌋ + 1] - a[⌊b[i]⌋]) * (b[i] - ⌊b[i]⌋)
	@discardableResult
	public mutating func vLinearInterpolation<C : BufferCollection, D : BufferCollection>(of a:C, by b:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		vDSP_vlint(a.buffer(), b.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, b.vCount)), vDSP_Length(a.vCount)) ; return self
	}
	
	//	MARK: -
	
	/// - Returns: self sorted
	@discardableResult
	public mutating func vSort(descending:Bool = false) -> Self {
		vDSP_vsort(mutableBuffer(), vDSP_Length(vCount), descending ? -1 : 1) ; return self
	}
	
	/// - Returns: self reversed
	@discardableResult
	public mutating func vReverse() -> Self {
		vDSP_vrvrs(mutableBuffer(), 1, vDSP_Length(vCount)) ; return self
	}
	
	/// - Returns: self[i] = 0
	@discardableResult
	public mutating func vClear() -> Self {
		vDSP_vclr(mutableBuffer(), 1, vDSP_Length(vCount)) ; return self
	}
	
	/// - Returns: self[i] = value
	@discardableResult
	public mutating func vFill(with value:Float) -> Self { var value = value
		vDSP_vfill(&value, mutableBuffer(), 1, vDSP_Length(vCount)) ; return self
	}
	
	/// - Returns: self[i] = lower + (upper - lower) * i / n
	@discardableResult
	public mutating func vFill(with lower:Float, to upper:Float) -> Self { var lower = lower, upper = upper
		vDSP_vgen(&lower, &upper, mutableBuffer(), 1, vDSP_Length(vCount)) ; return self
	}
	
	/// - Returns: self[i] = value + step * i
	@discardableResult
	public mutating func vFill(with value:Float, step:Float) -> Self { var value = value, step = step
		vDSP_vramp(&value, &step, mutableBuffer(), 1, vDSP_Length(vCount)) ; return self
	}
	
	/// - Returns: self[i] = a[i] * (value + step * i)
	@discardableResult
	public mutating func vRamp<C : BufferCollection>(of a:C, times value:Float, step:Float) -> (Self, Float) where C.BufferElement : IsFloat { var value = value, step = step
		vDSP_vrampmul(a.buffer(), 1, &value, &step, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount))) ; return (self, value)
	}
	
	/// - Returns: self[i] = a[i] * (value + step * i) + self[i]
	@discardableResult
	public mutating func vPlusRamp<C : BufferCollection>(of a:C, times value:Float, step:Float) -> (Self, Float) where C.BufferElement : IsFloat { var value = value, step = step
		vDSP_vrampmuladd(a.buffer(), 1, &value, &step, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount))) ; return (self, value)
	}
	
	/// - Returns: self[i] = a[i] * (1 - i/n) + b[i] * i/n
	@discardableResult
	public mutating func vMerge<C : BufferCollection, D : BufferCollection>(from a:C, to b:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		vDSP_vtmerg(a.buffer(), 1, b.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount))) ; return self
	}
	
	/// - Returns: self[i] = a[i]
	@discardableResult
	public mutating func vCopy<C : BufferCollection>(from a:C) -> Self where C.BufferElement : IsFloat {
		cblas_scopy(Int32(Swift.min(vCount, a.vCount)), a.buffer(), 1, mutableBuffer(), 1) ; return self
	}
	
	/// - Returns: self[i] = Float(a[i])
	@discardableResult
	public mutating func vCopy<C : BufferCollection>(from a:C) -> Self where C.BufferElement : IsDouble {
		vDSP_vdpsp(a.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount))) ; return self
	}
	
	/// - Returns: self[i] = Float(a[i])
	@discardableResult
	public mutating func vCopy<C : BufferCollection>(from a:C) -> Self where C.BufferElement == Int32 {
		vDSP_vflt32(a.rawPointer().assumingMemoryBound(to:Int32.self), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.count))) ; return self
	}
	
	/// - Returns: self[i] = 0.42 - 0.5 * cos(2π•i/n) + 0.08 * cos(4π•i/n)
	@discardableResult
	public mutating func vBlackmanWindow(half:Bool = false) -> Self {
		vDSP_blkman_window(mutableBuffer(), vDSP_Length(vCount), Int32(half ? vDSP_HALF_WINDOW : 0)) ; return self
	}
	
	/// - Returns: self[i] = 0.54 - 0.46 * cos(2π•i/n)
	@discardableResult
	public mutating func vHammingWindow(half:Bool = false) -> Self {
		vDSP_hamm_window(mutableBuffer(), vDSP_Length(vCount), Int32(half ? vDSP_HALF_WINDOW : 0)) ; return self
	}
	
	/// - Returns: self[i] = (normalized ? 0.8165 : 0.5) * (1 - cos(2π•i/n))
	@discardableResult
	public mutating func vHanningWindow(half:Bool = false, normalized:Bool = false) -> Self {
		vDSP_hann_window(mutableBuffer(), vDSP_Length(vCount), Int32(half ? vDSP_HALF_WINDOW : 0) | Int32(normalized ? vDSP_HANN_NORM : vDSP_HANN_DENORM)) ; return self
	}
	
	/// - Returns: self[i] = (a[i] - m) / d where m = ∑ a[i]/n, d = √(∑ a[i]²/n - (∑ a[i]/n)²)
	@discardableResult
	public mutating func vNormalizeDistribution<C : BufferCollection>(of a:C) -> (normalized:Self, mean:Float, deviation:Float) where C.BufferElement : IsFloat {
		var mean:Float = 0, deviation:Float = 0 ; vDSP_normalize(a.buffer(), 1, mutableBuffer(), 1, &mean, &deviation, vDSP_Length(Swift.min(vCount, a.vCount))) ; return (self, mean, deviation)
	}
	
	/// - Returns: self[i] = ∑ a[i + j] * b[j]
	@discardableResult
	public mutating func vConvolution<C : BufferCollection, D : BufferCollection>(of a:C, filter b:D, convolution flag:Bool = false) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		let length = b.vCount
		let kernel = flag ? b.buffer().advanced(by: length - 1) : b.buffer()
		let stride = flag ? -1 : 1
		vDSP_conv(a.buffer(), 1, kernel, stride, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount - length + 1)), vDSP_Length(length)) ; return self
	}
	
	/// - Returns: self[i] = ∑ a[i * factor + j] * b[j]
	@discardableResult
	public mutating func vFiniteImpulseResponse<C : BufferCollection, D : BufferCollection>(of a:C, filter b:D, decimation factor:Int) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		let length = b.vCount
		vDSP_desamp(a.buffer(), factor, b.buffer(), mutableBuffer(), vDSP_Length(Swift.min(vCount, (a.vCount - length) / factor + 1)), vDSP_Length(length)) ; return self
	}
	
	/// - Returns: self[i] = a[i]*b[0] + a[i-1]*b[1] + a[i-2]*b[2] - self[i-1]*b[3] - self[i-2]*b[4]
	@discardableResult
	public mutating func vDifferenceEquation22<C : BufferCollection, D : BufferCollection>(of a:C, coefficients b:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		vDSP_deq22(a.buffer(), 1, b.buffer(), mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount) - 2)) ; return self
	}
	
	/// - Returns: self[i] = (amplitude ? 20 : 10) * log10(a[i] / zeroReference)
	@discardableResult
	public mutating func vDecibels<C : BufferCollection>(of a:C, zeroReference:Float = 1, amplitude:Bool = false) -> Self where C.BufferElement : IsFloat {
		var b = zeroReference
		vDSP_vdbcon(a.buffer(), 1, &b, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount)), amplitude ? 1 : 0) ; return self
	}
	
	/// - Returns: self[i] <> a[i]
	@discardableResult
	public mutating func vSwap<C : MutableBufferCollection>(with a:inout C) -> Self where C.BufferElement : IsFloat {
		//cblas_sswap(Int32(length), array.mutableBuffer(), 1, mutableBuffer(), 1) ; return self
		vDSP_vswap(a.mutableBuffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount))) ; return self
	}
	
	/// - Returns: self[i] = a[i] - ⌊a[i]⌋
	@discardableResult
	public mutating func vFraction<C : BufferCollection>(of a:C) -> Self where C.BufferElement : IsFloat {
		vDSP_vfrac(a.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount))) ; return self
	}
	
	/// - Returns: self[i] = a[i] ± ε towards b[i]
	@discardableResult
	public mutating func vNext<C : BufferCollection, D : BufferCollection>(after a:C, towards b:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount, b.vCount)) ; vvnextafterf(mutableBuffer(), a.buffer(), b.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = a[i] with sign of b[i]
	@discardableResult
	public mutating func vCopy<C : BufferCollection, D : BufferCollection>(magnitude a:C, sign b:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		var length = Int32(Swift.min(vCount, a.vCount, b.vCount)) ; vvcopysignf(mutableBuffer(), b.buffer(), a.buffer(), &length) ; return self
	}
	
	/// - Returns: self[i] = a[indices[i]]
	@discardableResult
	public mutating func vGather<C : BufferCollection>(from a:C, indices:[vDSP_Length]) -> Self where C.BufferElement : IsFloat {
		vDSP_vgathr(a.buffer().advanced(by: 1), indices, 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, indices.count))) ; return self
	}
	
	/// - Returns: self[i] = a[indices[i] - 1]
	@discardableResult
	public mutating func vGather<C : BufferCollection>(from a:C, oneBased indices:[vDSP_Length]) -> Self where C.BufferElement : IsFloat {
		vDSP_vgathr(a.buffer(), indices, 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, indices.count))) ; return self
	}
	
	/// - Returns: self[i] = a[⌊indices[i]⌋]
	@discardableResult
	public mutating func vGather<C : BufferCollection, D : BufferCollection>(from a:C, indices b:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		vDSP_vindex(a.buffer(), b.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, b.vCount))) ; return self
	}
	
	/// - Returns: self[j] = a[j] where ++j if b[i] != 0
	@discardableResult
	public mutating func vCompress<C : BufferCollection, D : BufferCollection>(_ a:C, gate b:D) -> Self where C.BufferElement : IsFloat, D.BufferElement : IsFloat {
		vDSP_vcmprs(a.buffer(), 1, b.buffer(), 1, mutableBuffer(), 1, vDSP_Length(Swift.min(vCount, a.vCount, b.vCount))) ; return self
	}
}

//	MARK: -

extension BufferCollection where BufferElement : IsFloat {
	/// - Returns: maximum value
	public func vMaximum() -> Float {
		var result:Float = 0 ; vDSP_maxv(buffer(), 1, &result, vDSP_Length(vCount)) ; return result
	}
	
	/// - Returns: maximum value, index
	public func vMaximumIndex() -> (value:Float, index:Int) {
		var result:Float = 0, index:vDSP_Length = 0 ; vDSP_maxvi(buffer(), 1, &result, &index, vDSP_Length(vCount)) ; return (result, Int(index))
	}
	
	public func vMaximumIndex(in range:Range<Int>, backwards:Bool = false) -> (value:Float, index:Int) {
		var result:Float = 0, index:vDSP_Length = 0
		vDSP_maxvi(buffer().advanced(by:backwards ? range.upperBound - 1 : range.lowerBound), backwards ? -1 : 1, &result, &index, vDSP_Length(Swift.min(vCount, range.count)))
		return (result, (backwards ? range.count - 1 - Int(~index &+ 1) : Int(index)) + range.lowerBound)
	}
	
	/// - Returns: maximum magnitude
	public func vMaximumMagnitude() -> Float {
		var result:Float = 0 ; vDSP_maxmgv(buffer(), 1, &result, vDSP_Length(vCount)) ; return result
	}
	
	/// - Returns: maximum magnitude, index
	public func vMaximumMagnitudeIndex() -> (value:Float, index:Int) {
		var result:Float = 0, index:vDSP_Length = 0 ; vDSP_maxmgvi(buffer(), 1, &result, &index, vDSP_Length(vCount)) ; return (result, Int(index))
		//let index = cblas_isamax(Int32(vCount), buffer(), 1) ; return (self[index], index)
	}
	
	/// - Returns: minimum value
	public func vMinimum() -> Float {
		var result:Float = 0 ; vDSP_minv(buffer(), 1, &result, vDSP_Length(vCount)) ; return result
	}
	
	/// - Returns: minimum value, index
	public func vMinimumIndex() -> (value:Float, index:Int) {
		var result:Float = 0, index:vDSP_Length = 0 ; vDSP_minvi(buffer(), 1, &result, &index, vDSP_Length(vCount)) ; return (result, Int(index))
	}
	
	/// - Returns: minimum magnitude
	public func vMinimumMagnitude() -> Float {
		var result:Float = 0 ; vDSP_minmgv(buffer(), 1, &result, vDSP_Length(vCount)) ; return result
	}
	
	/// - Returns: minimum magnitude, index
	public func vMinimumMagnitudeIndex() -> (value:Float, index:Int) {
		var result:Float = 0, index:vDSP_Length = 0 ; vDSP_minmgvi(buffer(), 1, &result, &index, vDSP_Length(vCount)) ; return (result, Int(index))
	}
	
	/// - Parameter limit: Maximum number of sign changes to traverse before returning.
	/// - Returns: index of sign change if limit reached, number of sign changes.
	public func vZeroCrossing(limit:Int) -> (indexAfterCrossing:Int, crossings:Int) {
		let length = vCount
		var crossingCount:vDSP_Length = 0
		var index:vDSP_Length = 0
		
		if limit < 0 {
			vDSP_nzcros(buffer().advanced(by:length - 1), -1, vDSP_Length(-limit), &index, &crossingCount, vDSP_Length(length))
			
			if index != 0 { index = vDSP_Length(length) + index }
		} else {
			vDSP_nzcros(buffer(), 1, vDSP_Length(limit), &index, &crossingCount, vDSP_Length(length))
		}
		
		return (Int(index), Int(crossingCount))
	}
	
	/// - Returns: ∑ self[i] / n
	public func vMean() -> Float {
		var result:Float = 0 ; vDSP_meanv(buffer(), 1, &result, vDSP_Length(vCount)) ; return result
	}
	
	/// - Returns: ∑ |self[i]| / n
	public func vMeanMagnitude() -> Float {
		var result:Float = 0 ; vDSP_meamgv(buffer(), 1, &result, vDSP_Length(vCount)) ; return result
	}
	
	/// - Returns: ∑ (self[i]²) / n
	public func vMeanSquares() -> Float {
		var result:Float = 0 ; vDSP_measqv(buffer(), 1, &result, vDSP_Length(vCount)) ; return result
	}
	
	/// - Returns: ∑ (self[i] * |self[i]|) / n
	public func vMeanSignPreservingSquares() -> Float {
		var result:Float = 0 ; vDSP_mvessq(buffer(), 1, &result, vDSP_Length(vCount)) ; return result
	}
	
	/// - Returns: √(∑ (self[i]²) / n)
	public func vRootMeanSquare() -> Float {
		var result:Float = 0 ; vDSP_rmsqv(buffer(), 1, &result, vDSP_Length(vCount)) ; return result
	}
	
	/// - Returns: ∑ self[i]
	public func vSum() -> Float {
		var result:Float = 0 ; vDSP_sve(buffer(), 1, &result, vDSP_Length(vCount)) ; return result
	}
	
	/// - Returns: ∑ |self[i]|
	public func vSumMagnitudes() -> Float {
		var result:Float = 0 ; vDSP_svemg(buffer(), 1, &result, vDSP_Length(vCount)) ; return result
	}
	
	/// - Returns: ∑ (self[i]²)
	public func vSumSquares() -> Float {
		var result:Float = 0 ; vDSP_svesq(buffer(), 1, &result, vDSP_Length(vCount)) ; return result
	}
	
	/// - Returns: ∑ (self[i] * |self[i]|)
	public func vSumSignPreservingSquares() -> Float {
		var result:Float = 0 ; vDSP_svs(buffer(), 1, &result, vDSP_Length(vCount)) ; return result
	}
	
	/// - Returns: ∑ self[i], ∑ self[i]²
	public func vSums() -> (sum:Float, sumSquares:Float) {
		var sum:Float = 0, sumSquares:Float = 0 ; vDSP_sve_svesq(buffer(), 1, &sum, &sumSquares, vDSP_Length(vCount)) ; return (sum, sumSquares)
	}
	
	/// - Returns: ∑ self[i]/n, √(∑ self[i]²/n - (∑ self[i]/n)²)
	public func vDeviation() -> (mean:Float, deviation:Float) {
		var mean:Float = 0, deviation:Float = 0 ; vDSP_normalize(buffer(), 1, nil, 1, &mean, &deviation, vDSP_Length(vCount)) ; return (mean, deviation)
	}
	
	/// - Returns: ∑ self[i] * a[i]
	public mutating func vDot<C : BufferCollection>(times a:C) -> Float where C.BufferElement : IsFloat {
		var product:Float = 0 ; vDSP_dotpr(buffer(), 1, a.buffer(), 1, &product, vDSP_Length(Swift.min(vCount, a.vCount))) ; return product
		//return cblas_sdot(Int32(Swift.min(vCount, a.vCount)), buffer(), 1, a.buffer(), 1)
	}
	
	/// - Returns: ∑ (self[i] - a[i])²
	public func vDistanceSquared<C : BufferCollection>(to a:C) -> Float where C.BufferElement : IsFloat {
		var distanceSquared:Float = 0 ; vDSP_distancesq(buffer(), 1, a.buffer(), 1, &distanceSquared, vDSP_Length(Swift.min(vCount, a.vCount))) ; return distanceSquared
	}
	
	/// - Returns: √∑ self[i]²
	public func vNormal() -> Float {
		return cblas_snrm2(Int32(vCount), buffer(), 1)
	}
	
	//	MARK: -
	
	/// - Returns: indices sorted by values of self, so self[indices[0]] ... self[indices[n]] are sorted
	@discardableResult
	public func vSort<C : MutableBufferCollection>(indices a:inout C, descending:Bool = false) -> C where C.BufferElement == vDSP_Length {
		let length = vDSP_Length(Swift.min(vCount, a.count))
		let target = a.mutableRawPointer().assumingMemoryBound(to:vDSP_Length.self)
		for i in 0 ..< length { target[Int(i)] = i }
		vDSP_vsorti(buffer(), target, nil, length, descending ? -1 : 1)
		return a
	}
	
	/// - Returns: ⌊self[i]⌋ rounded towards zero
	public func vTruncated() -> [Int32] {
		let length = vCount
		var result = [Int32](repeating:0, count:length)
		vDSP_vfix32(buffer(), 1, &result, 1, vDSP_Length(length))
		return result
	}
	
	/// - Returns: ⌊self[i]⌋ rounded towards nearest, towards infinity when equidistant
	public func vRounded() -> [Int32] {
		let length = vCount
		var result = [Int32](repeating:0, count:length)
		vDSP_vfixr32(buffer(), 1, &result, 1, vDSP_Length(length))
		return result
	}
}

//	MARK: -

extension BufferCollection where BufferElement : IsFloat {
	public func vExtract(size:(major:Int, minor:Int), from:(major:Int, minor:Int), minor:Int) -> [BufferElement] {
		var result = [BufferElement](vCount:size.minor * size.major)
		return result.vAssign((self, minor), size:size, minor:size.minor, from:from).matrix
	}
	public func vNegative() -> [BufferElement] {
		var result = [BufferElement](zeros:count) ; return result.vNegative(of:self)
	}
	public func vPlus(_ a:Float) -> [BufferElement] {
		var result = [BufferElement](zeros:count) ; return result.vSum(of:self, plus:a)
	}
	public func vTimes(_ a:Float) -> [BufferElement] {
		var result = [BufferElement](zeros:count) ; return result.vProduct(of:self, times:a)
	}
	public func vDivided(by a:Float) -> [BufferElement] {
		var result = [BufferElement](zeros:count) ; return result.vQuotient(of:self, over:a)
	}
	public func vModulo(_ a:Float) -> [BufferElement] {
		var result = [BufferElement](zeros:count) ; result.vFill(with:a) ; return result.vRemainder(of:self, modulus:result)
	}
	public func vTo(power a:Float) -> [BufferElement] {
		var result = [BufferElement](zeros:count) ; return result.vRaise(self, toPower:a)
	}
	public func vPlus<C : BufferCollection>(_ a:C) -> [BufferElement] where C.BufferElement : IsFloat {
		var result = [BufferElement](vCount:Swift.min(vCount, a.vCount)) ; return result.vSum(of:self, plus:a)
	}
	public func vMinus<C : BufferCollection>(_ a:C) -> [BufferElement] where C.BufferElement : IsFloat {
		var result = [BufferElement](vCount:Swift.min(vCount, a.vCount)) ; return result.vDifference(of:self, minus:a)
	}
	public func vTimes<C : BufferCollection>(_ a:C) -> [BufferElement] where C.BufferElement : IsFloat {
		var result = [BufferElement](vCount:Swift.max(vCount, a.vCount)) ; return result.vProduct(of:self, times:a)
	}
	public func vDivided<C : BufferCollection>(by a:C) -> [BufferElement] where C.BufferElement : IsFloat {
		var result = [BufferElement](vCount:Swift.min(vCount, a.vCount)) ; return result.vQuotient(of:self, over:a)
	}
	public func vModulo<C : BufferCollection>(_ a:C) -> [BufferElement] where C.BufferElement : IsFloat {
		var result = [BufferElement](vCount:Swift.min(vCount, a.vCount)) ; return result.vRemainder(of:self, modulus:a)
	}
	public func vTo<C : BufferCollection>(power a:C) -> [BufferElement] where C.BufferElement : IsFloat {
		var result = [BufferElement](vCount:Swift.min(vCount, a.vCount)) ; return result.vRaise(self, toPower:a)
	}
	
	/// - Parameters:
	///   - major: The range of rows to access.
	///   - minor: The range of columns to access.
	///   - matrixMinor: The number of columns in the receiver.
	public subscript(major major:CountableRange<Int>, minor minor:CountableRange<Int>, matrixMinor:Int) -> [BufferElement] {
		get { return vExtract(size:(major:major.count, minor:minor.count), from:(major:major.lowerBound, minor:minor.lowerBound), minor:matrixMinor) }
	}
}

extension MutableBufferCollection where BufferElement : IsFloat {
	@discardableResult
	public mutating func vAdd(_ a:Float) -> Self {
		return vSum(of:self, plus:a)
	}
	@discardableResult
	public mutating func vMultiply(by a:Float) -> Self {
		return vProduct(of:self, times:a)
		//cblas_sscal(Int32(vCount), a, mutableBuffer(), 1)
	}
	@discardableResult
	public mutating func vDivide(by a:Float) -> Self {
		return vQuotient(of:self, over:a)
	}
	@discardableResult
	public mutating func vRaise(toPower a:Float) -> Self {
		return vRaise(self, toPower:a)
	}
	
	/// - Parameters:
	///   - major: The range of rows to access.
	///   - minor: The range of columns to access.
	///   - matrixMinor: The number of columns in the receiver.
	public subscript(major major:CountableRange<Int>, minor minor:CountableRange<Int>, matrixMinor:Int) -> [BufferElement] {
		get { return vExtract(size:(major:major.count, minor:minor.count), from:(major:major.lowerBound, minor:minor.lowerBound), minor:matrixMinor) }
		set { vAssign((newValue, minor.count), size:(major:major.count, minor:minor.count), minor:matrixMinor, at:(major:major.lowerBound, minor:minor.lowerBound)) }
	}
}

extension ResizableBufferCollection where BufferElement : IsFloat {
	@discardableResult
	public mutating func vAdd<C : BufferCollection>(_ a:C) -> Self where C.BufferElement : IsFloat {
		vExpand(capacity:a.vCount) ; return vSum(of:self, plus:a)
	}
	@discardableResult
	public mutating func vSubtract<C : BufferCollection>(_ a:C) -> Self where C.BufferElement : IsFloat {
		vExpand(capacity:a.vCount) ; return vDifference(of:self, minus:a)
	}
	@discardableResult
	public mutating func vMultiply<C : BufferCollection>(by a:C) -> Self where C.BufferElement : IsFloat {
		vExpand(capacity:a.vCount) ; return vProduct(of:self, times:a)
	}
	@discardableResult
	public mutating func vDivide<C : BufferCollection>(by a:C) -> Self where C.BufferElement : IsFloat {
		vExpand(capacity:a.vCount) ; return vQuotient(of:self, over:a)
	}
	@discardableResult
	public mutating func vRemainder<C : BufferCollection>(modulus a:C) -> Self where C.BufferElement : IsFloat {
		vExpand(capacity:a.vCount) ; return vRemainder(of:self, modulus:a)
	}
	@discardableResult
	public mutating func vRaise<C : BufferCollection>(toPower a:C) -> Self where C.BufferElement : IsFloat {
		vExpand(capacity:a.vCount) ; return vRaise(self, toPower:a)
	}
	@discardableResult
	public mutating func vAccumulateLinearAverage<C : BufferCollection>(of a:C, with prior:Float) -> Self where C.BufferElement : IsFloat {
		vExpand(capacity:a.vCount) ; return vLinearAverage(of:a, with:prior)
	}
}

//	MARK: -

extension BufferCollection where BufferElement : IsFloat {
	/// - Returns: string suitable for printf style formatting of a `BufferElement`
	public func vNumberFormat(precision:Int) -> (format:String, hasTrailingZeroes:Bool) {
		let maximum = vMaximumMagnitude()
		let width = maximum.isFinite ? Int(ceil(log10(maximum))) + 2 : 6
		
		if width > precision { return ("%\(precision).\(Swift.max(0, 7 - precision))e", false) }
		
		let a = buffer()
		for i in 0 ..< vCount where fabs(modf(a[i]).1) > 0 {
			return ("%\(precision).\(precision - 1 - width)f", true)
		}
		
		return ("%\(width).0f", false)
	}
	
	/// Generate a formatted textual representation of this collection as a matrix.
	///
	/// - Parameters:
	///   - size: Number of columns and rows in the matrix.
	///   - maximum: Number of columns and rows in the matrix to include in the description.
	///   - maximumWidth: Maximum character width of the description.
	/// - Returns: Description of the values in the collection as a matrix.
	public func vMatrixDescription(size:(major:Int, minor:Int), maximum:(major:Int, minor:Int) = (0, 0), maximumWidth:Int = 0) -> String {
		var result = ""
		
		let maximumMinor = Swift.max(maximum.minor > 0 ? Swift.min(maximum.minor, size.minor) : size.minor, 1)
		let maximumMajor = Swift.min(maximum.major > 0 ? Swift.min(maximum.major, size.major) : size.major, vCount / maximumMinor)
		let precision = maximumWidth > maximumMinor ? Swift.max(4, Swift.min(maximumWidth / maximumMinor, 12)) : 8
		let minorSeparator = ","
		let majorSeparator = maximumMinor < size.minor ? ", …\n" : "\n"
		let notZero = CharacterSet(charactersIn:"0").inverted
		let (format, trimZeros) = vNumberFormat(precision:precision)
		
		for major in 0 ..< maximumMajor {
			for minor in 0 ..< maximumMinor {
				let element = self[major * size.minor + minor]
				var string = String(format:format, element as! CVarArg)
				
				if trimZeros, let range = string.rangeOfCharacter(from:notZero, options:.backwards), range.upperBound < string.endIndex {
					let removeLength = string.distance(from:string.startIndex, to:(string[range.lowerBound] == "." ? range.lowerBound : range.upperBound))
					let stringLength = string.distance(from:string.startIndex, to:string.endIndex)
					string = string.padding(toLength:removeLength, withPad:"", startingAt:0).padding(toLength:stringLength, withPad:" ", startingAt:0)
				}
				
				result.append(string)
				result.append(minor + 1 == maximumMinor ? majorSeparator : minorSeparator)
			}
		}
		
		if maximumMajor < size.major {
			result.append("…\n")
		}
		
		return result
	}
}

//	MARK: -

public prefix func .- <C : BufferCollection>(rhs:C) -> [C.BufferElement] where C.BufferElement : IsFloat { return rhs.vNegative() }

public func .+ <C : BufferCollection, D : BufferCollection> (lhs:C, rhs:D) -> [C.BufferElement] where C.BufferElement : IsFloat, D.BufferElement : IsFloat { return lhs.vPlus(rhs) }
public func .+ <C : BufferCollection> (lhs:C, rhs:Float) -> [C.BufferElement] where C.BufferElement : IsFloat { return lhs.vPlus(rhs) }
public func .- <C : BufferCollection, D : BufferCollection> (lhs:C, rhs:D) -> [C.BufferElement] where C.BufferElement : IsFloat, D.BufferElement : IsFloat { return lhs.vMinus(rhs) }
public func .- <C : BufferCollection> (lhs:C, rhs:Float) -> [C.BufferElement] where C.BufferElement : IsFloat { return lhs.vPlus(-rhs) }
public func .* <C : BufferCollection, D : BufferCollection> (lhs:C, rhs:D) -> [C.BufferElement] where C.BufferElement : IsFloat, D.BufferElement : IsFloat { return lhs.vTimes(rhs) }
public func .* <C : BufferCollection> (lhs:C, rhs:Float) -> [C.BufferElement] where C.BufferElement : IsFloat { return lhs.vTimes(rhs) }
public func ./ <C : BufferCollection, D : BufferCollection> (lhs:C, rhs:D) -> [C.BufferElement] where C.BufferElement : IsFloat, D.BufferElement : IsFloat { return lhs.vDivided(by:rhs) }
public func ./ <C : BufferCollection> (lhs:C, rhs:Float) -> [C.BufferElement] where C.BufferElement : IsFloat { return lhs.vDivided(by:rhs) }
public func .% <C : BufferCollection, D : BufferCollection> (lhs:C, rhs:D) -> [C.BufferElement] where C.BufferElement : IsFloat, D.BufferElement : IsFloat { return lhs.vModulo(rhs) }
public func .^ <C : BufferCollection, D : BufferCollection> (lhs:C, rhs:D) -> [C.BufferElement] where C.BufferElement : IsFloat, D.BufferElement : IsFloat { return lhs.vTo(power:rhs) }
public func .^ <C : BufferCollection> (lhs:C, rhs:Float) -> [C.BufferElement] where C.BufferElement : IsFloat { return lhs.vTo(power:rhs) }

public func .+= <C : ResizableBufferCollection, D : BufferCollection> (lhs:inout C, rhs:D) where C.BufferElement : IsFloat, D.BufferElement : IsFloat { lhs.vAdd(rhs) }
public func .-= <C : ResizableBufferCollection, D : BufferCollection> (lhs:inout C, rhs:D) where C.BufferElement : IsFloat, D.BufferElement : IsFloat { lhs.vSubtract(rhs) }
public func .*= <C : ResizableBufferCollection, D : BufferCollection> (lhs:inout C, rhs:D) where C.BufferElement : IsFloat, D.BufferElement : IsFloat { lhs.vMultiply(by:rhs) }
public func ./= <C : ResizableBufferCollection, D : BufferCollection> (lhs:inout C, rhs:D) where C.BufferElement : IsFloat, D.BufferElement : IsFloat { lhs.vDivide(by:rhs) }
public func .%= <C : ResizableBufferCollection, D : BufferCollection> (lhs:inout C, rhs:D) where C.BufferElement : IsFloat, D.BufferElement : IsFloat { lhs.vRemainder(modulus:rhs) }
public func .^= <C : ResizableBufferCollection, D : BufferCollection> (lhs:inout C, rhs:D) where C.BufferElement : IsFloat, D.BufferElement : IsFloat { lhs.vRaise(toPower:rhs) }

public func .+= <C : MutableBufferCollection> (lhs:inout C, rhs:Float) where C.BufferElement : IsFloat { lhs.vAdd(rhs) }
public func .-= <C : MutableBufferCollection> (lhs:inout C, rhs:Float) where C.BufferElement : IsFloat { lhs.vAdd(-rhs) }
public func .*= <C : MutableBufferCollection> (lhs:inout C, rhs:Float) where C.BufferElement : IsFloat { lhs.vMultiply(by:rhs) }
public func ./= <C : MutableBufferCollection> (lhs:inout C, rhs:Float) where C.BufferElement : IsFloat { lhs.vDivide(by:rhs) }
public func .^= <C : MutableBufferCollection> (lhs:inout C, rhs:Float) where C.BufferElement : IsFloat { lhs.vRaise(toPower:rhs) }
