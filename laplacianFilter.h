//#pragma once //(not entirely portable, so use instead:)

#ifndef _LAPLACIANFILTER_H_
#define _LAPLACIANFILTER_H_

/*
 * This defines enums that increment and decrement, wrapping around if
 * necessary, using ++ and -- operators, for both the structuring
 * element types and the filter types.
 *
 * ——Ian Calegory, 12/20/2016
 */

//#include <map>
//std::map<StructuringElementEnum, std::string> elementNames;
//elementNames[disk3x3] = "3x3 Disk Structuring Element";
// Use array instead:

enum StructuringElementEnum
{
	disk3x3,
	disk5x5,
	disk7x7,
	square3x3,
	square5x5,
	square7x7,
	ring3x3,
	ring5x5,
	ring7x7
};

// From http://www.cplusplus.com/forum/beginner/41790/
// To support wrap-around incrementing/decrement
inline StructuringElementEnum& operator++(StructuringElementEnum& eDOW, int)  // <--- note -- must be a reference
{
	const int i = static_cast<int>(eDOW) + 1;
	eDOW = static_cast<StructuringElementEnum>((i) % 9);
	return eDOW;
}

inline StructuringElementEnum& operator--(StructuringElementEnum& type, int)  // <--- note -- must be a reference
{
	const int i = static_cast<int>(type) - 1;

	if (i < 0) // Check whether to cycle to last item if number goes below 0
	{
		type = static_cast<StructuringElementEnum>(8);
	}
	else // Else set it to current number -1
	{
		type = static_cast<StructuringElementEnum>((i) % 9);
	}
	return type;
}

/*
// This can be used with enum class: enum class StructuringElementEnum
// http://stackoverflow.com/questions/15450914/incrementation-and-decrementation-of-enum-class
// Increment the enum, wrapping around
StructuringElementEnum& operator++(StructuringElementEnum &e) {
	using IntType = typename std::underlying_type<StructuringElementEnum>::type;
	e = static_cast<StructuringElementEnum>(static_cast<IntType>(e) + 1);
	if (e == StructuringElementEnum::END_OF_LIST)
		e = static_cast<StructuringElementEnum>(0);
	return e;
}
// Increment the enum, wrapping around
StructuringElementEnum& operator--(StructuringElementEnum &e) {
	using IntType = typename std::underlying_type<StructuringElementEnum>::type;
	e = static_cast<StructuringElementEnum>(static_cast<IntType>(e) - 1);
	if (e == StructuringElementEnum::BEGINNING_OF_LIST)
		//e = static_cast<StructuringElementEnum>(StructuringElementEnum::END_OF_LIST) - 1;
		e = static_cast<StructuringElementEnum>(StructuringElementEnum::END_OF_LIST) - 1;
	return e;
}
*/

// Second const specifies internal linkage by default, which is why you can define them
// in header files without violating the One Definition Rule. (Otherwise linker produces
// an already defined in *.obj error message)
const char* const elementNames[]
{
	"3x3 Disk",
	"5x5 Disk",
	"7x7 Disk",
	"3x3 Square",
	"5x5 Square",
	"7x7 Square",
	"3x3 Ring",
	"5x5 Ring",
	"7x7 Ring"
};

enum FilterTypeEnum
{
	AlmostAReference,
	AlmostFlattened,
	AntiAliasingSmoothFuzz,
	FuzzInWideOutline,
	GhostEdges,
	InvisoWithWideOutlines,
	MosaicInGray,
	PsychedelicLines,
	PsychedelicMellowed,
	ReliefInGray
};

inline FilterTypeEnum& operator++(FilterTypeEnum& eDOW, int)
{
	const int i = static_cast<int>(eDOW) + 1;
	eDOW = static_cast<FilterTypeEnum>((i) % 10);
	return eDOW;
}

inline FilterTypeEnum& operator--(FilterTypeEnum& type, int) 
{
	const int i = static_cast<int>(type) - 1;

	if (i < 0) // Check whether to cycle to last item if number goes below 0
	{
		type = static_cast<FilterTypeEnum>(9);
	}
	else // Else set it to current number -1
	{
		type = static_cast<FilterTypeEnum>((i) % 10);
	}
	return type;
}

const char* const filterTypeNames[]
{
	"Almost A Reference",
	"Almost Flattened",
	"Anti-aliasing Smooth Fuzz [NOT IN CUDA VERSION FOR SOME REASON!]",
	"Fuzz In Wide Outline",
	"Ghost Edges",
	"Inviso With Wide Outlines",
	"Mosaic In Gray",
	"Psychedelic Lines",
	"Psychedelic Mellowed",
	"Relief In Gray"
};

#endif // #ifndef _LAPLACIANFILTER_H_