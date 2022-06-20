#!/usr/bin/env wolframscript
(* ::Package:: *)

<<TDG`;
low=-5;
high=20;
plot=LuescherPlot[{low,high}]


target=FileNameJoin[{DirectoryName@AbsoluteFileName[$InputFileName],"S2.pdf"}]
Export[target,plot]
