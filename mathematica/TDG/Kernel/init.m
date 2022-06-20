(* ::Package:: *)

(* ::Title:: *)
(*First*)


BeginPackage["TDG`"]


TDGLicense::usage="    TDG: solve reduced two-body quantum mechanics in the A1 representation in a square two-dimensional volume.
    Copyright (C) <year>  <name of author>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
";


hbarc::usage="1 in sensible units."
MeV::usage="1 in our units."
fm::usage="1/hbarc";
M::usage="938 MeV, the nucleon mass.";


Group::usage="A group; Group['name'][conjugacy classes]";
Class::usage="A conjugacy class of group elements; Class['name'][{matrix representations}][<|irreps -> characters|>]";
D4::usage="A Group instance for the D4 group.";
representations::usage="Extracts the representations from a Group";

characterTable::usage="Character Table.  Takes one Group object.";
representationProduct::usage="Table of products of representations.";

(* Oh specifics *)
nsqVectors::usage="Representative vectors with a given continuum kinetic integer^2";
shell::usage="Partners in a given shell";


states::usage="states[nx]: generate a list of A1 states for a given nx.";
statesInShell::usage="statesInShell[nx][{x,y}]
For a square spatial lattice with nx sites on a side, counts the number of momentum partners of {x,y}.";


sites::usage="sites[nx]: generate integer coordinates for a square lattice with nx sites to a side.

For odd nx the coordinates are symmetric around 0.
For even nx the coordinates have an extra positive value.
"

coordinates::usage="coordinates[L, nx]: the physical coordinates for a square lattice with nx sites to a side of physical length L.

For odd nx the coordinates are symmetric around 0.
For even nx the coordinates have an extra positive value.
"

fundamentalDomain::usage="fundamentalDomain[nx]: a list of integer coordinates that, 
when all square symmetry operations are applied, generate all the coordinates in sites[nx]."


A1ToSites::usage="A1ToSites[nx][\[Psi]A1]
     nx    number of sites to a side
     \[Psi]A1    a list of numbers giving a wavefunction in A1 momentum space;
             the basis is given by states[nx]

Gives a wavefunction in the basis given by sites[nx].";
A1Project::usage="A1Project[nx][\[Psi]x]
     nx    number of sites to a side
     \[Psi]x    a list of numbers giving a wavefunction in position space;
             the basis is given by sites[nx]

Projects to the A1 momentum basis given by states[nx]
";


SpatialVisualize::usage="SpatialVisualize[nx][\[Psi]x]
     nx    number of sites to a side
     \[Psi]x    a list of numbers giving a wavefunction in position space;
             the basis is given by sites[nx]

At each spatial site there will be a sphere whose 
     radius corresponds to the magnitude of the wavefunction,
     hue corresponds to the phase.
";
SpatialVisualizeA1::usage="SpatialVisualize[nx][\[Psi]A1]
     nx    number of sites to a side
     \[Psi]A1    a list of numbers giving a wavefunction in A1 momentum space;
             the basis is given by states[nx]

At each spatial site (centered on the origin) there will be a sphere whose 
     radius corresponds to the magnitude of the wavefunction,
     hue corresponds to the phase.
";
VisualizeA1::usage="Visualize[nx][\[Psi]A1]
     nx    number of sites to a side
     \[Psi]A1    a list of numbers giving a wavefunction in A1 momentum space;
             the basis is given by states[nx]

At each allowed momentum in the Brillouin zone (centered on zero momentum) there will be a sphere whose 
     radius corresponds to the magnitude of the wavefunction,
     hue corresponds to the phase.
";


T::usage="T[L, nx, \[Mu]]
     L     physical extent of the square volume
     nx    number of sites to a side
     \[Mu]     reduced mass
For a square spatial lattice of physical extent L with nx sites on a side,
a matrix in the basis given by states[nx] which is all zero but for the diagonal, given by \!\(\*SuperscriptBox[\(p\), \(2\)]\)/2\[Micro] for a reduced mass \[Mu].
";

LegoSphereOperator::usage="LegoSphereOperator[nx,r]
     nx     Integer
     r       triplet of integers {x,y}

For a square spatial lattice with nx sites on a side,
a matrix in the basis given by states[nx] with matrix elements given by
    \[LeftAngleBracket] A1 n' | V | A1 n \[RightAngleBracket] = 1/Sqrt[NN'] \[CapitalSigma][g,g'] \[LeftAngleBracket] g' n' | V | g n \[RightAngleBracket]
where V depends on r, which dictates the stencil of the 'Lego Sphere',
	V[x,y] = \[CapitalSigma][R] \[Delta][x, y+Rr],
the rotations g,g', and R are summed over the Oh group,
N and N' are the normalizations of | A1 n \[RightAngleBracket] and | A1 n' \[RightAngleBracket] (given by statesInShell).
";

H::usage="H[L,nx,\[Mu],cs]
     L      Physical extent of the square volume
     nx    number of sites to a side
     \[Mu]      reduced mass
     cs    an Association from LegoSphere radius to Wilson coefficient
             <|r -> c |>
             example: <|{0,0,0} -> 0.567, {0,1,1} -> 1.234|>

The two-body (reduced mass \[Mu]) center-of-mass momentum=0 Hamiltonian for a box of length L with nx sites,
H=   Kinetic     = \!\(\*SuperscriptBox[\(p\), \(2\)]\)/2\[Micro]
     + Potential = \[CapitalSigma][r] c[r]/\!\(\*SuperscriptBox[\(L\), \(3\)]\) LegoSphereOperator[nx, r]
where the sum over r lets us pass one or more specification of a LegoSphere and its amplitude.
";

interacting::usage="interacting[nx]

When there is more than one shell per non-interacting energy the diagonalization of H yields
one interacting state and the rest with non-interacting energies.  We can just skip those energies and get the 'truly' interacting ones.

interacting[nx] gives a list of indices that give interacting energy eigenvalues when H is diagonalized and the eigenvalues sorted.
"


LuescherZeta::usage="LuescherZeta[qtsq]
     qtsq     Numerical dimensionless scattering momentum squared, 
                  'q tilde squared'
                  qtsq = \!\(\*SuperscriptBox[OverscriptBox[\(q\), \(~\)], \(2\)]\) = (qL/2\[Pi]\!\(\*SuperscriptBox[\()\), \(2\)]\) = 2\[Mu]E(L/2\[Pi]\!\(\*SuperscriptBox[\()\), \(2\)]\)

Normalized so that L q cot \[Delta](q) = 2/\!\(\*SqrtBox[\(\[Pi]\)]\) LuescherZeta[(qL/2\[Pi]\!\(\*SuperscriptBox[\()\), \(2\)]\)]
or equivalently           \!\(\*OverscriptBox[\(q\), \(~\)]\) cot \[Delta](q) = \!\(\*FractionBox[\(1\), SuperscriptBox[\(\[Pi]\), \(3/2\)]]\)LuescherZeta[\!\(\*SuperscriptBox[OverscriptBox[\(q\), \(~\)], \(2\)]\)]

Only numerical arguments are accepted.";

LuescherZetaPoles::usage="LuescherZetaPoles[max]
     max      List all the real arguments of the LuescherZeta that diverge.
";

LuescherPlot::usage=""

LuescherTuningTargetEnergy::usage="LuescherTuningTargetEnergy[L,\[Mu],ere][level]
     L      Physical extent of the square volume
     \[Mu]      reduced mass
     ere   effective range expansion; a function of (dimensionful) \!\(\*SuperscriptBox[\(momentum\), \(2\)]\).
             For example, if you've got a scattering length a and an effective range r (and no further shape parameters)
             you could define
                  ere[a_,r_][qsq_]:=-\!\(\*FractionBox[\(1\), \(a\)]\)+\!\(\*FractionBox[\(1\), \(2\)]\)r qsq
             and evaluate
                  LuescherTuningTargetEnergy[L,\[Mu],ere[a,r]][1]
             to compute the lowest finite-volume energy level for that scattering length and effective range.
             Similarly,
                  LuescherTuningTargetEnergy[L,\[Mu],ere[a,r]] /@ Range[1,10]
             would compute the 10 lowest finite-volume energy levels.

The units had better make sense!";


Begin["Private`"]


(* ::Title::Closed:: *)
(*License*)


TDGLicense[]:=Print[TDGLicense::usage]


(* ::Title::Closed:: *)
(*Parallel Setup*)


If[$KernelCount > 0,
	TABLE=ParallelTable; MAP=ParallelMap,
	TABLE=Table; MAP=Map
];


(* ::Title::Closed:: *)
(*Choice of units*)


(* ::Text:: *)
(*You could pick anything consistent here.*)
(*I pick units that match Neill's exactly.*)


(* ::Input::Initialization:: *)
hbarc=197.;
MeV=1;
fm = 1/hbarc;
M = 938 MeV;(*  NOTE!  My M is the nucleon mass but in my Hamiltonian above I use the REDUCED mass \[Mu] = M/2! *)


(* ::Text:: *)
(*NOTE:  In the Hamiltonian I picked the parameters as \[Mu], THE REDUCED MASS.*)
(*FOR \[Mu] YOU ALMOST CERTAINLY WANT M/2*)


(* ::Title::Closed:: *)
(*Verbose Printing*)


verbose=False;
VPrint[header_][message___]:=If[verbose,Print[header,": ",message]]


(* ::Title::Closed:: *)
(*Long-Term Memoization*)


TDGLibrary=FileNameJoin[{
	DirectoryName@AbsoluteFileName[FindFile["TDG/Kernel/init.m"]],
	"storage"}];


Storage[nx_]:=Storage[nx]=FileNameJoin[{
	TDGLibrary,
	"nx="<>ToString[nx]
	}]

StorageFile[nx_][name_]:=Module[{},
	CreateStorage[nx];
	Return@FileNameJoin[{Storage[nx],name<>".mx"}]
	]
	
CreateStorage[nx_]:=Quiet@CreateDirectory[Storage[nx]]

StoreTo[nx_][name_][value_]:=Module[
	{file=StorageFile[nx][name]},
	VPrint["TDG internal"]["Saving to    ",file];Export[file,value];
	value
]

RetrieveFrom[nx_][name_]:=Import[StorageFile[nx][name]]


SetAttributes[ComputeOnce,HoldAllComplete]

(* 
    This trick lets us write a curried

        ComputeOnce[nx][name][expression] 

    and have Hold work as necessary.
    
    For details see the discussion at https://stackoverflow.com/a/11561797/
*)
ComputeOnce[a__]:=Function[x,ComputeOnce[a,x],HoldAll]

ComputeOnce[nx_,name_,expression_]:=With[
	{file=StorageFile[nx][name]},
	If[
		FileExistsQ[file],
		VPrint["TDG internal"]["Reading from ",file]; Import[file],
		StoreTo[nx][name][expression]
	]
]


(* ::Title::Closed:: *)
(*D4 Group Theory*)


(* ::Section::Closed:: *)
(*Comments*)


(* ::Text:: *)
(*You don't have to look in here.*)


(* ::Text:: *)
(*Most of this infrastructure I built up for a different project where I might have needed other groups / the little groups of Oh given P!=0.  So if this seems very ineloquent, it's because the infrastructure was designed to handle much more than our case here.*)
(**)
(*https://www.webqc.org/symmetrypointgroup-d4.html*)


(* ::Section::Closed:: *)
(*Group Theory Setup*)


(* ::Input::Initialization:: *)
(*Class[name_][operations__][characters_]*)
name[Class[name_][__][_]]:=name
operations[Class[_][operations__][_]]:={operations}
characters[Class[_][__][characters_]]:=characters


(* ::Input::Initialization:: *)
(*Group[name_][classes__]*)
name[Group[name_][__]]:=name
classes[Group[_][classes__]]:=classes
operations[Group[_][classes__]]:=Join@@(operations/@{classes})
representations[Group[_][classes__]]:=Union[Keys/@characters/@{classes}][[1]]
closedQ[Group[_][classes__]]:=With[{o=operations[OH]},Table[MemberQ[o,g . h],{g,o},{h,o}]==ConstantArray[True,{Length[o],Length[o]}]]

characterTable[group_]:=With[
{C={classes[group]}},
{names=(name/@C)},
{R=representations[group]},
{{""}~Join~names}~Join~Table[{r,Sequence@@Table[r/.characters[c],{c,C}]},{r,R}]
]


(* ::Section::Closed:: *)
(*D4*)


identity=Class["E"][IdentityMatrix[2]][<|"A1"->1,"A2"->1,"B1"->1,"B2"->1,"E"->2|>];
quarterRotations=Class["2C4(z)"][
	RotationMatrix[+(2\[Pi])/4],
	RotationMatrix[-(2\[Pi])/4]
	][<|"A1"->1,"A2"->1,"B1"->-1,"B2"->-1,"E"->0|>];
halfRotation=Class["C2(z)"][
	RotationMatrix[+(2\[Pi])/2](* Also the inversion *)
	][<|"A1"->1,"A2"->1,"B1"->1,"B2"->1,"E"->2|>];
edgeReflection=Class["2C'2"][
	+PauliMatrix[3](* fix x, flip y *),
	-PauliMatrix[3](* flip x, fix y *)
	][<|"A1"->1,"A2"->-1,"B1"->1,"B2"->-1,"E"->0|>];
cornerReflection=Class["2C''2"][
	+PauliMatrix[1](* fixes x=+y *),
	-PauliMatrix[1](* fixes x=-y *)
	][<|"A1"->1,"A2"->-1,"B1"->-1,"B2"->1,"E"->0|>];
	
D4=Group["D4"][
	identity,quarterRotations,halfRotation,edgeReflection,cornerReflection];


(* ::Input::Initialization:: *)
representationProduct[group_,compress_:True][reps__]:=With[{c=characterTable[group]},
With[{r=Map[Function[rep,Rest[Select[c,#[[1]]==rep&][[1]]]],{reps}]},
With[{product=Times@@r},
	With[{multiplicities=multiplicity/@c[[2;;,1]]},
	With[{linearCombo=multiplicities . c[[2;;,2;;]]},
	With[{decomposition=Solve[linearCombo==product,multiplicities][[1]]},
	With[{solution=decomposition/.(multiplicity->Identity)},
		If[Not[compress],
			solution,
			Total[solution/.((rep_->copies_):>copies rep)]
		]/.Plus->CirclePlus
]]]]]]]


(* ::Section::Closed:: *)
(*Representative vectors*)


(* ::Input::Initialization:: *)
nsqVectors[nsq_]:=PowersRepresentations[nsq,2,2]
image[group_][vector_]:=Union[Table[o . vector,{o,operations[D4]}]]
representative[group_][vector_]:=Last[image[group][vector]]
shell[vector_]:=shell[vector]=image[D4][vector]


(* ::Title::Closed:: *)
(*States*)


(* ::Text:: *)
(*Because we know both the kinetic and the potential are A1 we know that the states are partitioned not only by their parity but even by their Subscript[D, 4] irrep.*)
(**)
(*We can use this to reduce our required work, as long as we are careful.  We do not need to operate on all the momenta.*)
(**)
(*For example, there is one n^2=1 state,*)
(*	|A1 01 \[RightAngleBracket] = (\[VerticalSeparator]0 1\[RightAngleBracket] + |1 0\[RightAngleBracket] + \[VerticalSeparator]0 \[Dash]1\[RightAngleBracket] + |\[Dash]1 0\[RightAngleBracket])/Sqrt[4]*)


(* ::Input::Initialization:: *)
states[nx_]:=states[nx]=(*ComputeOnce[nx]["states"]@ *)Select[Flatten[Join[nsqVectors/@Range[0,2((nx+1)/2)^2]],1],AllTrue[LessThan[(nx+1)/2]]]


(* ::Text:: *)
(*We're going to need to count how many states are on each shell.*)


(* ::Input::Initialization:: *)
statesInShell[nx_][vector_]:=statesInShell[nx][vector]=With[{boundary=Floor[(nx+1)/2]}, Switch[vector,
(* The boundary counting makes it work for both even and odd nx *)
{0,0},1,
{0,y_},4,
{x_,x_},4,
{x_,y_},8
]/2^Count[vector,boundary]
]


(* ::Title::Closed:: *)
(*Sites*)


(* ::Text:: *)
(*We can restrict our attention to the fundamental domain of the spatial lattice.*)
(*The fundamental domain is the set of points that, when all possible symmetry operations are applied, generate the whole lattice.*)
(*For a square lattice the fundamental domain is much smaller than the whole square of points.*)
(*The shape of the fundamental domain is a right triangle with vertices at*)
(*	(0,0)*)
(*	(0,n)*)
(*	(n,n)*)
(*	with n = nx/2*)
(*which, when reflected and rotated according to D4 (modulo the periodic boundary conditions) generates all the sites.*)
(*The size of the fundamental domain is the same size as the set of momentum states in the D4 sector.*)
(**)
(*In fact, the list of integer coordinates and the lattice momenta are precisely the same!*)


fundamentalDomain=states


(* ::Text:: *)
(*We can enumerate the integer coordinates in a square lattice very simply*)


sites[nx_]:=sites[nx]=Flatten[shell/@states[nx],1]


(* ::Text:: *)
(*We can turn site labels into coordinates*)


coordinates[L_,nx_?OddQ]:=L sites[nx]/(nx-1)
coordinates[L_, nx_?EvenQ]:=L sites[nx]/nx


(* ::Title::Closed:: *)
(*Changing Basis*)


(* ::Text:: *)
(*In the A1 basis a state can be described by a set of amplitudes and kets*)
(*	| \[CapitalPsi] \[RightAngleBracket] = \[Sum](p) \[Psi](p) | A1 p \[RightAngleBracket]*)


(* ::Text:: *)
(*To transform into position space simply apply*)
(*	1 = \[Sum](x) | x \[RightAngleBracket]\[LeftAngleBracket] x|:*)
(*	*)
(*	|\[CapitalPsi] \[RightAngleBracket] = \[Sum](x) | x \[RightAngleBracket]\[LeftAngleBracket] x| \[Sum](p) \[Psi](p) | A1 p \[RightAngleBracket]*)
(*	        = \[Sum](x) [ \[Sum](p) \[Psi](p) \[LeftAngleBracket] x | A1 p \[RightAngleBracket] ] | x \[RightAngleBracket]*)
(**)
(*The only thing to note is that | A1 p \[RightAngleBracket] isn't one momentum state but is a normalized superposition of momentum states given by applying A1 rotations to p.*)
(*For example,*)
(*	|A1 01 \[RightAngleBracket] = (\[VerticalSeparator]0 1\[RightAngleBracket] + |1 0\[RightAngleBracket] + \[VerticalSeparator]0 \[Dash]1\[RightAngleBracket] + |\[Dash]1 0\[RightAngleBracket])/Sqrt[4]*)


A1ToSitesSlow[nx_][\[Psi]A1_]:=Table[
	1/Sqrt[nx^2] (* <-- unitary convention for the fourier transform *) Total[
		MapThread[
			Function[{amplitude,momentum},
				(* This function computes the terms in the inner square brackets in the sum above, \[Psi](p) \[LeftAngleBracket] x | A1 p \[RightAngleBracket] *)
				With[{momenta=shell[momentum]},
					(* Multiply the amplitude by the expanded normalized A1 state in position space. *)
					amplitude/Sqrt[Length[momenta]] ExpToTrig[Sum[Exp[+ 2 \[Pi] I p . r/nx],{p,momenta}]]
					]
			],
			{\[Psi]A1, states[nx]}
			]
		],
	{r,sites[nx]}]


(* ::Text:: *)
(*Rather than using this brute-force method, however, we can use our knowledge that the state is in A1.*)
(*Our strategy is to just find the amplitudes in the fundamental domain and then use the symmetry operations to build out the amplitude for every site.*)


A1ToSites[nx_][\[Psi]A1_]:=Flatten@Table[
	ConstantArray[(* copy the amplitude to each site from its preimage in the fundamental domain *)
		1/Sqrt[nx^2] (* <-- unitary convention for the fourier transform *)Total[
			MapThread[
				Function[{amplitude,momentum},
					With[{momenta=shell[momentum]},
						(* This function computes the terms in the inner square brackets in the sum above, \[Psi](p) \[LeftAngleBracket] x | A1 p \[RightAngleBracket] *)
						(* Multiply the amplitude by the expanded normalized A1 state in position space. *)
						amplitude/Sqrt[Length[momenta]] ExpToTrig[Sum[Exp[+ 2 \[Pi] I p . r/nx],{p,momenta}]]
					]
				],
				{\[Psi]A1, states[nx]}
			]
		],
		Length[shell@r]],
	{r,states[nx]}]


(* ::Text:: *)
(*To go from position space to A1 space requires a projection,*)
(*	P = \[Sum](p) | A1 p \[RightAngleBracket]\[LeftAngleBracket] A1 p |*)
(* since a state in position space*)
(* 	| \[CapitalPsi] \[RightAngleBracket] = \[Sum](x) \[Psi](x) | x \[RightAngleBracket]*)
(* we can have contributions from other representations.*)


(* ::Text:: *)
(*Simply apply P*)
(*	P| \[CapitalPsi] \[RightAngleBracket] = \[Sum](p) | A1 p \[RightAngleBracket]\[LeftAngleBracket] A1 p | \[Sum](x) \[Psi](x) | x \[RightAngleBracket]*)
(*		= \[Sum](p) [ \[Sum](x) \[Psi](x) \[LeftAngleBracket] A1 p |  x \[RightAngleBracket] ] | A1 p \[RightAngleBracket]*)
(*		*)
(*Note that in the implementation below we do NOT assume that the position-space \[Psi] is purely A1!*)


A1Project[nx_][\[Psi]x_]:=Table[
	With[{momenta = shell[p]},
		1/Sqrt[nx^2] (* <-- unitary convention for the fourier transform *) Total[
			MapThread[
				Function[{amplitude,x},
				(* This function computes the terms in the inner square brackets in the sum above, \[Psi](x) \[LeftAngleBracket] A1 p | x \[RightAngleBracket] *)
				(* Multiply the amplitude by the expanded normalized A1 state in position space. *)
				(* The - in the exponent, since A1ToSites has + *)
					amplitude/Sqrt[Length[momenta]] ExpToTrig[Sum[Exp[- 2 \[Pi] I k . x / nx],{k,momenta}]]
				],
				{\[Psi]x,sites[nx]}
			]
		]
	],
	{p,states[nx]}]


(* ::Title::Closed:: *)
(*Visualization*)


(* ::Text:: *)
(*We can visualize an arbitrary wavefunction in position space.*)


SpatialVisualize[nx_][\[Psi]x_]:=Graphics[
		MapThread[
			Function[{\[Psi],x},{Hue[0.5+Arg[\[Psi]]/(2\[Pi])],Disk[x,Abs[\[Psi]]]}],
			{
				\[Psi]x,
				sites[nx]
			}
		]
]


(* ::Text:: *)
(*We can transform an A1 momentum-basis wavefunction and visualize it in position space.*)


SpatialVisualizeA1[nx_][\[Psi]A1_]:=SpatialVisualize[nx][A1ToSites[nx][\[Psi]A1]]


(* ::Text:: *)
(*We can also visualize in momentum space*)


VisualizeA1[nx_][\[Psi]A1_]:=Graphics[
		MapThread[
			Function[{\[Psi],A1p},With[{normalization=1/Sqrt[statesInShell[nx][A1p]]},
				Table[{Hue[0.5+Arg[\[Psi]]/(2\[Pi])],Disk[p,normalization Abs[\[Psi]]]},{p,shell[A1p]}]]
				],
			{
				\[Psi]A1,
				states[nx]
			}
		]
]


(* ::Title:: *)
(*Kinetic*)


(* ::Text:: *)
(*Let's write down the dimensionless kinetic Hamiltonian.*)
(*For |\[Psi]\[RightAngleBracket] = |xy\[RightAngleBracket] in momentum space the kinetic hamiltonian is diagonal.  \[LeftAngleBracket] i | T | j \[RightAngleBracket] = 1/2 ((2\[Pi])/Nx)^2 Subscript[n, i]^2 Subscript[\[Delta], ij]*)
(**)
(*Given a set of states we just need to know the number of sites*)


(* ::Input::Initialization:: *)
pSquared[nx_]:=(*ComputeOnce[nx]["pSquared"]@*)DiagonalMatrix@Map[Dot[#,#]&,states[nx]]
T[nx_]:=2(* <-- reduced mass *) 1/2  ((2\[Pi])/nx)^2 pSquared[nx]


(* ::Title::Closed:: *)
(*A1 Matrix Element \[Dash] Slow but Obvious*)


(* ::Text:: *)
(*The LegoSphere simplifies in some situations (for example, when r' = 0) but let's just code it as-is.*)


Clear[LegoSphereOperatorSlow]
LegoSphereOperatorSlow[nx_, R_]:=LegoSphereOperatorSlow[nx,R]=With[{\[Psi]=states[nx]},
	TABLE[
		1/Sqrt[statesInShell[nx][m]statesInShell[nx][np]] 1/(nx^2 statesInShell[nx][R]) Sum[Exp[2\[Pi] I (gpnp-gn) . hr / nx],{gpnp,shell[m]},{gn,shell[np]}, {hr,shell[R]}],
		{m, \[Psi]},{np, \[Psi]}]
];


(* ::Title::Closed:: *)
(*A1 Lego Sphere Matrix Element Accelerated*)


rString[rprime_]:=StringJoin[Riffle[ToString/@Abs[rprime],"-"]]


(* ::Input::Initialization:: *)
LegoSphereOperator[nx_,R_]:=LegoSphereOperator[nx,R]=(*ComputeOnce[nx][rString[R]]@*)With[
{\[Psi]=states[nx]},
With[{
		norm=1/nx^2 Outer[Times,#,#]&@(1/Sqrt[MAP[statesInShell[nx],\[Psi]]]),
		expdot=MAP[Exp[2\[Pi] I shell[#] . R/nx]&,\[Psi]]
	},
	norm*TABLE[Total[Outer[Times,m,n],2],{m,expdot},{n,Conjugate[expdot]}]
]]


(* ::Text:: *)
(*We could save another factor of 2 by just computing the upper-triangle and using hermiticity.*)


(* ::Title::Closed:: *)
(*H in the A1g Sector*)


(* ::Text:: *)
(*Now we need to construct the Hamiltonian in the A1 sector,*)
(*	H | 0, n \[RightAngleBracket] = Subscript[H, 0] | 0, n \[RightAngleBracket] +  Subscript[\[Sum], r'] Subscript[C, r'] LegoSphere(r', X) | 0, n \[RightAngleBracket]*)
(*where we have one term in the potential for each radius we use (and I changed the basis to [dimensionless] momentum space labels).*)


(* ::Text:: *)
(*In our HPC code we have r'={0,0,0} and r'={0,1,1}.*)


(* ::Input::Initialization:: *)
Clear[H];
H[nx_,cs_?(AllTrue[NumericQ]@*Values)]:=H[nx,cs]= (
T[nx]+Total[KeyValueMap[ #2 LegoSphereOperator[nx,#1]&,cs]]
)



(* ::Text:: *)
(*When there is more than one shell per non-interacting energy the diagonalization of H yields one interacting state and the rest with non-interacting energies.*)


shellsWithDegeneracies[nx_]:=With[{\[Psi]=states[nx]},
	Tally[Dot[#,#]&/@\[Psi]]
]
interacting[nx_]:=interacting[nx]=Rest@FoldList[{#1[[1]]+#1[[2]],#2[[2]]}&,{1,0},shellsWithDegeneracies[nx]][[All,1]]


(* ::Title:: *)
(*L\[UDoubleDot]scher*)


(* ::Section:: *)
(*Numerical Implementation of LuescherZeta*)


(* ::Text:: *)
(*S2 as given in K\[ODoubleDot]rber, Berkowitz, and Luu*)
(*The real part is given in (34).*)
(*To convert to cot \[Delta], don't forget the imaginary pieces is described in (32 / 33)*)


S2Cutoff[cutoff_:200]:=Function[x,Evaluate[Sum[SquaresR[2,nsquared]/(nsquared-x),{nsquared,0,(cutoff/2)^2}]-2\[Pi] Log[cutoff/2]]];
S2=S2Cutoff[200];
LuescherZeta=S2;


(* ::Subsection:: *)
(*Bound \!\(\*OverscriptBox[\(q\), \(~\)]\)^2 to be on the right domain of the L\[UDoubleDot]scher zeta*)


LuescherZetaPoles[max_]:=Select[Range[0,Ceiling[max]],SquaresR[2,#]!=0&]
LuescherZetaDomain[level_]:=({-\[Infinity]}~Join~Select[Range[0,2 level],SquaresR[2,#]>0&])[[{level,level+1}]]

(* Nonzero \[Epsilon] prevents FindRoot from trying to evaluate LuescherZeta at one of its poles. *)
LuescherZetaDomainBound[level_,\[Epsilon]_:0.0001]:=If[level==1,{-1.,-\[Infinity],0},With[{d=LuescherZetaDomain[level]},{N@Mean[d]}~Join~d]]+{0,\[Epsilon],-\[Epsilon]}


(* ::Section:: *)
(*Plot*)


Options[LuescherPlot]={
	FrameLabel->{"x","\!\(\*FractionBox[\(1\), SuperscriptBox[\(\[Pi]\), \(2\)]]\)\!\(\*SubscriptBox[\(S\), \(2\)]\)(x)"}
}
LuescherPlot[{min_,max_},OptionsPattern[]]:=Block[{x},Plot[
	Evaluate@(LuescherZeta[x]/\[Pi]^2),{x,min,max},
	GridLines->{LuescherZetaPoles[max],{0}},
	FrameLabel->OptionValue[FrameLabel]
]]


(* ::Section:: *)
(*Matching the ERE and the Quantization Condition*)


(* ::Text:: *)
(*We use the dimensionless ere where \!\(\*OverscriptBox[\(a\), \(~\)]\) and Subscript[\!\(\*OverscriptBox[\(r\), \(~\)]\), e] appear, as well as the dimensionless x*)


(*ere[a_,re2_][x_]:= 2/\[Pi] (Log[a Sqrt[x]/2] + \[Gamma]) + 1/4 re2 x *)  (* you can add/parameterize higher powers, as long as you wind up with a function of q with a leading log. *)


(* ::Subsection:: *)
(*Match*)


LuescherTuningTargetEnergy[nx_,ere_][level_]:=Block[{x},
	(2\[Pi])^2 x /(nx^2)/.Chop@FindRoot[
		ere[x]-2/\[Pi] Log[Sqrt[x]]==1/\[Pi]^2 LuescherZeta[x],
		{x}~Join~LuescherZetaDomainBound[level],
		MaxIterations->10000
	]
]


(* ::Title:: *)
(*Finally*)


End[(* "Private`" *)]
EndPackage[]
