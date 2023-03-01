#!/usr/bin/env wolframscript
(* ::Package:: *)

<<TDG`;

a=1;
ERE[a_,re2_][x_]:=2/\[Pi] Log[a Sqrt[x]]+ re2 x
ere = ERE[a,0];


e[nx_,ere_,levels_:10]:=e[nx,ere,levels]=LuescherTuningTargetEnergy[nx,ere]/@Range[levels];
x[nx_,ere_,levels_:10]:=e[nx,ere,levels]/(2\[Pi])^2;
\[Zeta][nx_,ere_,levels_:10]:=1/\[Pi]^2 (LuescherZeta/@x[nx,ere,levels]);


zero[nx_,ere_][c0_?NumericQ,c1_?NumericQ]:=Sort[Eigenvalues[H[nx,<|{0,0}->c0,{0,1}->c1|>]]][[;;2]]-e[nx,ere,2]

tune[nx_,ere_]:=tune[nx,ere]=Block[{c0,c1},{c0,c1}/.FindRoot[
	zero[nx,ere][c0,c1],
	{c0,-3},
	{c1,+1}
]]

C0[nx_,ere_:ere]:=tune[nx,ere][[1]]
C1[nx_,ere_:ere]:=tune[nx,ere][[2]]
Energies[nx_,ere_:ere]:=Sort[Eigenvalues[H[nx,<|{0,0}->C0[nx,ere],{0,1}->C1[nx,ere]|>]]]
X[nx_,ere_:ere]:=Energies[nx,ere]/(2\[Pi])^2
Z[nx_,ere_:ere]:=(LuescherZeta/@X[nx,ere])/\[Pi]^2


min=7;
max=23;
NX=Range[min,max,2];
colors=Hue[(#-min)/(max-min)]&/@NX;
data={#,Splice[tune[#,ere]]}&/@NX


plotCoupling=ListPlot[
	List/@data[[All,{2,3}]],
	GridLines->{{0},{0}},
	PlotRange->{{Floor[Min[data[[All,2]]]-0.5],0.5},{-0.2,Ceiling[Max[data[[All,3]]]]}},
	PlotStyle->colors,
	Epilog->{
		Text[Style["  \!\(\*SubscriptBox[\(N\), \(x\)]\)="<>ToString[min],Large],data[[+1,{2,3}]],Left],
		Text[Style["  \!\(\*SubscriptBox[\(N\), \(x\)]\)="<>ToString[max],Large],data[[-1,{2,3}]],Left]
		},
	FrameLabel->{"\!\(\*SubscriptBox[\(\!\(\*OverscriptBox[\(C\), \(~\)]\)\), \(0\)]\)","\!\(\*SubscriptBox[\(\!\(\*OverscriptBox[\(C\), \(~\)]\)\), \((0, 1)\)]\)"}]


viewMinx=-3;
viewMaxx=+25;
viewMiny=-1;
viewMaxy=+1;

plotAmplitude=Show[
	LuescherPlot[{viewMinx,viewMaxx},FrameLabel->{"x","cot \[Delta] - \!\(\*FractionBox[\(2\), \(\[Pi]\)]\)log\!\(\*SqrtBox[\(x\)]\) = \!\(\*FractionBox[\(1\), SuperscriptBox[\(\[Pi]\), \(2\)]]\)\!\(\*SubscriptBox[\(S\), \(2\)]\)(x)"}],
	Plot[(ere[x]-2/\[Pi] Log[Sqrt[x]]),{x,viewMinx,viewMaxx},PlotStyle->{Black}],
	ListPlot[Evaluate@Table[Take[{X[nx],Z[nx]}\[Transpose],UpTo[15]],{nx,NX}],PlotStyle->colors],
	PlotRange->{{viewMinx,viewMaxx},{viewMiny,viewMaxy}}
] 


convergenceCoupling=FileNameJoin[{DirectoryName@AbsoluteFileName[$InputFileName],"improved-coupling.pdf"}]
convergenceAmplitude=FileNameJoin[{DirectoryName@AbsoluteFileName[$InputFileName],"improved-amplitude.pdf"}]
Export[convergenceCoupling,plotCoupling]
Export[convergenceAmplitude,plotAmplitude]
