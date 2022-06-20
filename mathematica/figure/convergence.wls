#!/usr/bin/env wolframscript
(* ::Package:: *)

<<TDG`;

a=1;
ERE[a_,re2_][x_]:=2/\[Pi] (Log[a/2 Sqrt[x]]+EulerGamma)+1/4 re2 x
ere = ERE[a,0];


e[nx_,ere_,levels_:10]:=e[nx,ere,levels]=LuescherTuningTargetEnergy[nx,ere]/@Range[levels];
x[nx_,ere_,levels_:10]:=e[nx,ere,levels] nx^2/(2\[Pi])^2;
\[Zeta][nx_,ere_,levels_:10]:=1/\[Pi]^2 (LuescherZeta/@x[nx,ere,levels]);


zero[nx_,ere_][c0_?NumericQ]:=Sort[Eigenvalues[H[nx,<|{0,0}->c0|>]]][[1]]-e[nx,ere,1][[1]]
tune[nx_,ere_]:=Block[{c0},c0/.FindRoot[
	zero[nx,ere][c0],
	{c0,1}
]]

C0[nx_,ere_:ere]:=C0[nx,ere]=tune[nx,ere]
Energies[nx_,ere_:ere]:=Sort[Eigenvalues[H[nx,<|{0,0}->C0[nx,ere]|>]]]
X[nx_,ere_:ere]:=Energies[nx,ere] nx^2/(2\[Pi])^2
Z[nx_,ere_:ere]:=(LuescherZeta/@X[nx,ere])/\[Pi]^2


min=5;
max=45;
NX=Range[min,max,2];
colors=Hue[(#-min)/(max-min)]&/@NX;
data={#,C0@#}&/@NX;
exact={#,-((2\[Pi])/Log[1/2 a Exp[EulerGamma]#])}&/@NX;


plotCoupling=GraphicsColumn[{
	Show[
		Plot[-((2\[Pi])/Log[1/2 a Exp[EulerGamma]Nx]),{Nx,min,max}],
		ListPlot[data],
		FrameLabel->{None,"\!\(\*SubscriptBox[\(C\), \(0\)]\)"}
	],
	ListPlot[{data[[All,1]],(data/exact)[[All,2]]}\[Transpose],FrameLabel->{"\!\(\*SubscriptBox[\(N\), \(x\)]\)","-\!\(\*FractionBox[SubscriptBox[\(C\), \(0\)], \(2  \[Pi]\)]\) log \!\(\*FractionBox[SuperscriptBox[\(e\), \(\[Gamma]\)], \(2\)]\)\!\(\*SubscriptBox[\(N\), \(x\)]\)\!\(\*OverscriptBox[\(a\), \(~\)]\)"}]
	}]



viewMinx=-3;
viewMaxx=+25;
viewMiny=-1;
viewMaxy=+1

plotAmplitude=Show[
	LuescherPlot[{viewMinx,viewMaxx},FrameLabel->{"x","cot \[Delta] - \!\(\*FractionBox[\(2\), \(\[Pi]\)]\)log\!\(\*SqrtBox[\(x\)]\) = \!\(\*FractionBox[\(1\), SuperscriptBox[\(\[Pi]\), \(2\)]]\)\!\(\*SubscriptBox[\(S\), \(2\)]\)(x)"}],
	Plot[(ere[x]-2/\[Pi] Log[Sqrt[x]]),{x,viewMinx,viewMaxx},PlotStyle->{Black}],
	ListPlot[Evaluate@Table[Take[{X[nx],Z[nx]}\[Transpose],UpTo[15]],{nx,NX}],PlotStyle->colors],
	PlotRange->{{viewMinx,viewMaxx},{viewMiny,viewMaxy}}
] 


convergenceCoupling=FileNameJoin[{DirectoryName@AbsoluteFileName[$InputFileName],"convergence-coupling.pdf"}]
convergenceAmplitude=FileNameJoin[{DirectoryName@AbsoluteFileName[$InputFileName],"convergence-amplitude.pdf"}]
Export[convergenceCoupling,plotCoupling]
Export[convergenceAmplitude,plotAmplitude]
