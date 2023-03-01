#!/usr/bin/env wolframscript
(* ::Package:: *)

<<TDG`;

a=2;
ERE[a_,re2_][x_]:=2/\[Pi] (Log[a Sqrt[x]])+re2 x
ere = ERE[a,0];


e[nx_,ere_,levels_:10]:=e[nx,ere,levels]=LuescherTuningTargetEnergy[nx,ere]/@Range[levels];
x[nx_,ere_,levels_:10]:=e[nx,ere,levels]/(2\[Pi])^2;
\[Zeta][nx_,ere_,levels_:10]:=1/\[Pi]^2 (LuescherZeta/@x[nx,ere,levels]);


zero[nx_,ere_][c0_?NumericQ]:=Sort[Eigenvalues[H[nx,<|{0,0}->c0|>]]][[1]]-e[nx,ere,1][[1]]
tune[nx_,ere_]:=Block[{c0},c0/.FindRoot[
	zero[nx,ere][c0],
	{c0,1}
]]

C0[nx_,ere_:ere]:=C0[nx,ere]=tune[nx,ere]
Energies[nx_,ere_:ere]:=Sort[Eigenvalues[H[nx,<|{0,0}->C0[nx,ere]|>]]]
X[nx_,ere_:ere]:=Energies[nx,ere]/(2\[Pi])^2
Z[nx_,ere_:ere]:=(LuescherZeta/@X[nx,ere])/\[Pi]^2


min=7;
max=23;
NX=Range[min,max,2];
colors=Hue[(#-min)/(max-min)]&/@NX;
data={#,C0@#}&/@NX;
exact={#,-((2\[Pi])/(Log[1/2 a #]+Log[2]-2/\[Pi] Catalan))}&/@NX;


plotCoupling=GraphicsColumn[{
	Show[
		Plot[-((2\[Pi])/(Log[1/2 a /oneOverNx]+Log[2]-2/\[Pi] Catalan )),{oneOverNx,0,1/min}],
		ListPlot[List/@({1/data[[All,1]],data[[All,2]]}\[Transpose]),PlotStyle->colors],
		FrameLabel->{None,"\!\(\*SubscriptBox[\(\!\(\*OverscriptBox[\(C\), \(~\)]\)\), \(0\)]\)"},
		PlotRange->{{-0.01,1/min+0.01},All}
	],
	ListPlot[
		List/@({1/data[[All,1]],(data/exact)[[All,2]]}\[Transpose]),
		PlotStyle->colors,
		PlotRange->{{-0.01,1/min+0.01},All},
		FrameLabel->{
			"1/\!\(\*SubscriptBox[\(N\), \(x\)]\)",
			"-\!\(\*FractionBox[SubscriptBox[OverscriptBox[\(C\), \(~\)], \(0\)], \(2  \[Pi]\)]\)(log\!\(\*OverscriptBox[\(a\), \(~\)]\)\!\(\*SubscriptBox[\(N\), \(x\)]\)-\!\(\*FractionBox[\(2\), \(\[Pi]\)]\)G)"
			}
		]
	}]



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


reliable = Table[Energies[nx,ere][[;;-2]],{nx,NX}];
reliable = Map[#[[1;;Min[Length/@reliable]]]&,reliable]\[Transpose];

plotEnergy=Show[
ListPlot[
	Table[{1/NX^2,r/r[[1]]}\[Transpose],{r,reliable}]
	],
FrameLabel->{"1/\!\(\*SuperscriptBox[SubscriptBox[\(N\), \(x\)], \(2\)]\)","\!\(\*SubscriptBox[\(\!\(\*OverscriptBox[\(E\), \(~\)]\)\), SubscriptBox[\(N\), \(x\)]]\)/\!\(\*SubscriptBox[\(\!\(\*OverscriptBox[\(E\), \(~\)]\)\), \(7\)]\)"}
]


reliableX = Table[X[nx,ere][[;;-2]],{nx,NX}];
reliableX = Map[#[[1;;Min[Length/@reliableX]]]&,reliableX]\[Transpose];

plotx=Show[
ListPlot[
	Table[{1/NX^2,r/r[[1]]}\[Transpose],{r,reliableX}]
	],
FrameLabel->{
	"1/\!\(\*SuperscriptBox[SubscriptBox[\(N\), \(x\)], \(2\)]\)",
	"\!\(\*SubscriptBox[\(x\), SubscriptBox[\(N\), \(x\)]]\)/\!\(\*SubscriptBox[\(x\), \(7\)]\)"
	}
]


convergenceCoupling=FileNameJoin[{DirectoryName@AbsoluteFileName[$InputFileName],"convergence-coupling.pdf"}]
convergenceAmplitude=FileNameJoin[{DirectoryName@AbsoluteFileName[$InputFileName],"convergence-amplitude.pdf"}]
convergenceEnergies=FileNameJoin[{DirectoryName@AbsoluteFileName[$InputFileName],"convergence-energies.pdf"}]
convergencex=FileNameJoin[{DirectoryName@AbsoluteFileName[$InputFileName],"convergence-x.pdf"}]
Export[convergenceCoupling,plotCoupling]
Export[convergenceAmplitude,plotAmplitude]
Export[convergenceEnergies,plotEnergy]
Export[convergencex,plotx]
