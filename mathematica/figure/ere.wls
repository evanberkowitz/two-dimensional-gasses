#!/usr/bin/env wolframscript
(* ::Package:: *)

<<TDG`;

a=1;
re2=0.125;

ERE[a_,re2_][x_]:=2/\[Pi] (Log[a/2 Sqrt[x]]+EulerGamma)+1/4 re2 x


nx=5;
e1=LuescherTuningTargetEnergy[nx,ERE[a,re2]]/@Range[10];
x1=e1 nx^2/(2\[Pi])^2;
\[Zeta]1=1/\[Pi]^2 (LuescherZeta/@x1);

nx=11;
e2=LuescherTuningTargetEnergy[nx,ERE[a,0]]/@Range[10];
x2=e2 nx^2/(2\[Pi])^2;
\[Zeta]2=1/\[Pi]^2 (LuescherZeta/@x2);


plot=Show[
	LuescherPlot[{-3,16},FrameLabel->{"x","cot \[Delta] - \!\(\*FractionBox[\(2\), \(\[Pi]\)]\)log\!\(\*SqrtBox[\(x\)]\) = \!\(\*FractionBox[\(1\), SuperscriptBox[\(\[Pi]\), \(2\)]]\)\!\(\*SubscriptBox[\(S\), \(2\)]\)(x)"}],
	Plot[(ERE[a,re2][x]-2/\[Pi] Log[Sqrt[x]]),{x,-3,16},PlotStyle->{Darker@Green}],
	Plot[(ERE[a,0][x]-2/\[Pi] Log[Sqrt[x]]),{x,-3,16},PlotStyle->{Purple}],
	ListPlot[{{x1,\[Zeta]1}\[Transpose],{x2,\[Zeta]2}\[Transpose]},PlotStyle->{Darker@Green,Purple}],
	PlotRange->{-1,1}
]


target=FileNameJoin[{DirectoryName@AbsoluteFileName[$InputFileName],"ere.pdf"}]
Export[target,plot]
