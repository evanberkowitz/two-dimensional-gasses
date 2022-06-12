#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Input:: *)
(**)


S2Cutoff[cutoff_:200]:=Function[x,Evaluate[Sum[SquaresR[2,nsquared]/(nsquared-x),{nsquared,0,(cutoff/2)^2}]-2\[Pi] Log[cutoff/2]]];
S2=S2Cutoff[200];


low=-5;
high=20;
poles=Select[Range[0,high],SquaresR[2,#]!=0&];
plot=Plot[
	Evaluate@S2[x],{x,-5,20},
	GridLines->{poles,{0}},
	FrameLabel->{"x","\!\(\*SubscriptBox[\(S\), \(2\)]\)(x)"}
	]


target=FileNameJoin[{DirectoryName@AbsoluteFileName[$InputFileName],"S2.pdf"}]
Export[target,plot]
