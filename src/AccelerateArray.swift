//
//  AccelerateArray.swift
//  MAS1
//
//  Created by Eric Cole on 2/28/17.
//  Copyright © 2017 Balance Software. All rights reserved.
//

import Accelerate

public protocol AccelerableElement { static var vNumbersPerElement:Int { get } ; static var zero:Self { get } }
public protocol IsFloat : AccelerableElement {}
public protocol IsDouble : AccelerableElement {}

//	MARK: -

extension Float : IsFloat { public static let vNumbersPerElement = 1 ; public static let zero = Float(0) }
extension Double : IsDouble { public static let vNumbersPerElement = 1 ; public static let zero = Double(0) }

//	MARK: -

public protocol BufferCollection : Collection {
	associatedtype BufferElement
	var count:Int { get }
	subscript(_:Int) -> BufferElement { get }
	
	func rawPointer() -> UnsafeRawPointer!
}

//	MARK: -

public protocol MutableBufferCollection : BufferCollection {
	subscript(_:Int) -> BufferElement { get set }
	mutating func mutableRawPointer() -> UnsafeMutableRawPointer!
}

//	MARK: -

public protocol ResizableBufferCollection : MutableBufferCollection {
	mutating func append<C : Collection>(contentsOf collection:C) where C.Iterator.Element == BufferElement
	mutating func reserveCapacity(_ minimumCapacity:Int)
}

//	MARK: -

extension Array : ResizableBufferCollection {
	public typealias BufferElement = Element
	
	public func rawPointer() -> UnsafeRawPointer! { return UnsafeRawPointer(self) }
	public mutating func mutableRawPointer() -> UnsafeMutableRawPointer! { return UnsafeMutableRawPointer(&self) }
}

extension Array where Element : AccelerableElement {
	init(zeros:Int) { self.init(repeating:Element.zero, count:zeros) }
	init(vCount:Int) { let n = Element.vNumbersPerElement ; self.init(zeros:(vCount + n - 1) / n) }
}

extension Array where Element : FloatingPoint {
	init(numbers count:Int, _ value:Element = Element.nan) { self.init(repeating:value, count:count) }
}

//	MARK: -

extension ArraySlice : ResizableBufferCollection {
	public typealias BufferElement = Element
	
	public func rawPointer() -> UnsafeRawPointer! { return withUnsafeBytes { $0.baseAddress } }
	public mutating func mutableRawPointer() -> UnsafeMutableRawPointer! { return withUnsafeMutableBytes { $0.baseAddress } }
}

//	MARK: -

extension UnsafeBufferPointer : BufferCollection {
	public typealias BufferElement = Element
	
	public func rawPointer() -> UnsafeRawPointer! { return UnsafeRawPointer(baseAddress) }
	
	subscript(range:Range<Int>) -> UnsafeBufferPointer { return UnsafeBufferPointer(start:baseAddress?.advanced(by:range.lowerBound), count:range.count) }
}

//	MARK: -

extension UnsafeMutableBufferPointer : MutableBufferCollection {
	public typealias BufferElement = Element
	
	public func rawPointer() -> UnsafeRawPointer! { return UnsafeRawPointer(baseAddress) }
	public mutating func mutableRawPointer() -> UnsafeMutableRawPointer! { return UnsafeMutableRawPointer(baseAddress) }
	
	subscript(range:Range<Int>) -> UnsafeMutableBufferPointer { return UnsafeMutableBufferPointer(start:baseAddress?.advanced(by:range.lowerBound), count:range.count) }
}

//	MARK: -

extension BufferCollection where BufferElement : AccelerableElement {
	public var vCount:Int { return count * BufferElement.vNumbersPerElement }
}

//	MARK: -

extension ResizableBufferCollection where BufferElement : AccelerableElement {
	public mutating func vExpand(capacity:Int, placeholder:BufferElement = BufferElement.zero) {
		let perElement = BufferElement.vNumbersPerElement
		let capacity = (capacity + perElement - 1) / perElement 
		let length = count
		guard length < capacity else { return }
		reserveCapacity(capacity)
		append(contentsOf:repeatElement(placeholder, count:capacity - length))
	}
}

//	MARK: -

extension RangeReplaceableCollection where Index == Int, IndexDistance == Int {
	/// Treat collections as two dimensional matrices and replace a range of elements.
	///
	/// If the elememts in this collection are in row major order, then this replaces a range of rows and
	/// minorCount is the number of columns in both collections.  Rows may be removed or inserted if the
	/// number of new rows does not equal the number of rows being replaced.  Replacing with an empty collection
	/// removes rows and replacing an empty range inserts rows.
	///
	/// - Parameters:
	///   - range: The range to replace. In row major order this is a range of rows.
	///   - collection: The new elements to insert. The number of elements should be a multiple of minorCount or trailing elements will not be inserted.
	///   - minorCount: The number of elements in the minor direction. For row major order this is the number of columns.
	public mutating func replaceMajor<C : BidirectionalCollection>(_ range:CountableRange<Int>, with collection:C, minorCount:Int) where C.Iterator.Element == Iterator.Element, C.SubSequence.Iterator.Element == Iterator.Element, C.IndexDistance == Int {
		guard minorCount > 0 else { return }
		
		let count = self.count
		let major = count / minorCount
		let lower = Swift.min(Swift.max(0, range.lowerBound), major) * minorCount
		let upper = range.upperBound < major ? range.upperBound * minorCount : count
		let extra = collection.count % minorCount
		let range:Range = lower ..< upper
		
		if extra > 0 {
			replaceSubrange(range, with:collection.dropLast(extra))
		} else {
			replaceSubrange(range, with:collection)
		}
	}
	
	/// Treat collections as two dimensional matrices and replace a range of elements.
	///
	/// If the elememts in this collection are in row major order, then this replaces a range of columns and
	/// minorCount is the number of columns in both collections.  Columns may be removed or inserted if the
	/// number of new columns does not equal the number of columns being replaced.  Replacing with an empty
	/// collection removes columns and replacing an empty range inserts columns.
	///
	/// - Parameters:
	///   - range: The range to replace. In row major order this is a range of columns.
	///   - collection: The new elements to insert. The number of elements should be a multiple of majorCount or trailing elements will not be inserted.
	///   - minorCount: The number of elements in the minor direction. For row major order this is the number of columns before replacement.
	public mutating func replaceMinor<C : BidirectionalCollection>(_ range:CountableRange<Int>, with collection:C, minorCount:Int) where C.Iterator.Element == Iterator.Element, C.SubSequence.Iterator.Element == Iterator.Element, C.Index == Int, C.IndexDistance == Int {
		guard minorCount > 0 else { return }
		
		let rangeCount = range.count
		let majorCount = count / minorCount
		let valueCount = collection.count / majorCount
		var majorIndex = majorCount
		
		if valueCount > rangeCount {
			reserveCapacity(majorCount * (minorCount + valueCount - rangeCount))
		}
		
		while majorIndex > 0 {
			majorIndex -= 1
			
			let replaceRange:Range = majorIndex * minorCount + range.lowerBound ..< majorIndex * minorCount + range.upperBound
			
			if valueCount > 0 {
				let collectionRange:Range = majorIndex * valueCount ..< (majorIndex + 1) * valueCount
				
				replaceSubrange(replaceRange, with:collection[collectionRange])
			} else {
				removeSubrange(replaceRange)
			}
		}
	}
}

//	MARK: - Operators

prefix operator .-

infix operator .+ : AdditionPrecedence
infix operator .- : AdditionPrecedence
infix operator .* : MultiplicationPrecedence
infix operator ./ : MultiplicationPrecedence
infix operator .% : MultiplicationPrecedence
infix operator .^ : MultiplicationPrecedence

infix operator .+= : AssignmentPrecedence
infix operator .-= : AssignmentPrecedence
infix operator .*= : AssignmentPrecedence
infix operator ./= : AssignmentPrecedence
infix operator .%= : AssignmentPrecedence
infix operator .^= : AssignmentPrecedence

public func |= <C : RangeReplaceableCollection, D : BidirectionalCollection>(lhs:inout C, rhs: D) where D.Iterator.Element == C.Iterator.Element, D.SubSequence.Iterator.Element == C.Iterator.Element, C.Index == Int, C.IndexDistance == Int, D.Index == Int, D.IndexDistance == Int {
	let major = rhs.count
	let minor = lhs.count / major
	lhs.replaceMinor(minor ..< minor, with:rhs, minorCount:minor)
}

public func |= <C : RangeReplaceableCollection, D : BidirectionalCollection>(lhs:inout C, rhs: (matrix:D, minor:Int)) where D.Iterator.Element == C.Iterator.Element, D.SubSequence.Iterator.Element == C.Iterator.Element, C.Index == Int, C.IndexDistance == Int, D.Index == Int, D.IndexDistance == Int {
	let major = rhs.matrix.count / rhs.minor
	let minor = lhs.count / major
	lhs.replaceMinor(minor ..< minor, with:rhs.matrix, minorCount:minor)
}

public func += <C : RangeReplaceableCollection, D : BidirectionalCollection>(lhs:inout C, rhs: (matrix:D, minor:Int)) where D.Iterator.Element == C.Iterator.Element, D.SubSequence.Iterator.Element == C.Iterator.Element, C.Index == Int, C.IndexDistance == Int, D.Index == Int, D.IndexDistance == Int {
	let major = lhs.count / rhs.minor
	lhs.replaceMajor(major ..< major, with:rhs.matrix, minorCount:rhs.minor)
}

//	MARK: -

extension DSPComplex : IsFloat { public static let vNumbersPerElement = 2 ; public static let zero = DSPComplex() }
extension DSPDoubleComplex : IsDouble { public static let vNumbersPerElement = 2 ; public static let zero = DSPDoubleComplex() }

extension __CLPK_complex : IsFloat { public static let vNumbersPerElement = 2 ; public static let zero = __CLPK_complex() }
extension __CLPK_doublecomplex : IsDouble { public static let vNumbersPerElement = 2 ; public static let zero = __CLPK_doublecomplex() }

#if arch(x86_64) || arch(arm64) // CGFLOAT_IS_DOUBLE
extension CGFloat : IsDouble { public static let vNumbersPerElement = 1 ; public static let zero = CGFloat(0) }
extension CGVector : IsDouble { public static let vNumbersPerElement = 2 }
extension CGPoint : IsDouble { public static let vNumbersPerElement = 2 }
extension CGSize : IsDouble { public static let vNumbersPerElement = 2 }
extension CGRect : IsDouble { public static let vNumbersPerElement = 4 }
#else
extension CGFloat : IsFloat { public static let vNumbersPerElement = 1 ; public static let zero = CGFloat(0) }
extension CGVector : IsFloat { public static let vNumbersPerElement = 2 }
extension CGPoint : IsFloat { public static let vNumbersPerElement = 2 }
extension CGSize : IsFloat { public static let vNumbersPerElement = 2 }
extension CGRect : IsFloat { public static let vNumbersPerElement = 4 }
#endif

extension IsFloat { public var vNumbersPerElement:Int { return MemoryLayout<Self>.size / MemoryLayout<Float>.size } }
extension IsDouble { public var vNumbersPerElement:Int { return MemoryLayout<Self>.size / MemoryLayout<Double>.size } }
