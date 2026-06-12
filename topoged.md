A Topological Encoder Decoder Framework for
Temporal Graph Learning
AnonymousAuthor(s)
Abstract
Mosttemporalgraphlearningmethodsreducefuturepredictiontodiscriminative
1
inference over past interactions, focusing on edges or node labels for a largely
2
persistentsetofnodes. Thisviewbreaksdownwhennewnodesandedgesappear,
3
since the future graph becomes a new object with changing size, composition,
4
andstructure. Weaddressthisgapbyframingtemporalgraphpredictionasan
5
inverse topology problem. Instead of predicting edges directly, we first predict
6
a multiscale topological descriptor of the future graph and then reconstruct a
7
plausiblefuturesnapshotthatrealizesthisdescriptorunderinductiveconstraints.
8
Thisapproachmakesglobalstructureandnodechurnexplicitpredictiontargetsand
9
producesforecastedgraphsonwhichdownstreamtaskscanbeevaluatedwithout
10
11
retraining.Across14temporalgraphdatasets,weevaluateourmethod,TOPOGED,
onnode,linkandgraphpropertypredictionandcompareitagainststate-of-the-art
12
13
temporalgraphmodels. TOPOGEDachievesasignificantimprovementinnode-
level forecasting accuracy over the strongest baseline, with a macro average of
14
0.49versus0.05. Italsooutperformsbaselinesin60%ofgraph-structuremetric
15
evaluations and yields a substantial increase in macro-average edge prediction
16
metrics, from near-zero to 0.12. Our results show that topology-guided graph
17
forecastingcanpredictinductivefuturesnapshotswhosestructuresupportsmultiple
18
downstreamevaluations.
19
1 Introduction
20
Dynamicgraphs,inwhichnodesandedgesevolveovertime,arecentraltomodelingrealsystemssuch
21
associalnetworks,citationgraphs,onlinecommunities,andblockchaintransactionnetworks[50,8].
22
Inthesesettings,theobjectofinterestisoftennotasinglefutureedgeorlabel,butthenextnetwork
23
stateitself. ArisksignalforanERC20token,astressindicatorforatransactionecosystem,ora
24
moderationstatisticforanonlineforumdependsonthestructureofthefuturegraph: whichnodes
25
remainactive,whichnewnodesarrive,andhowconnectivityreorganizes.
26
Most temporal graph learning methods approach this problem through discriminative prediction.
27
Theylearntemporalrepresentationsfrompastinteractionsandusethemtoscorecandidateedges
28
ornodelabels[45,42,31]. Thisparadigmiseffectivewhenthefuturegraphisassumedtobean
29
extensionofthepastoveralargelypersistentnodeset. However,thisassumptionbreaksdownunder
30
nodechurn. Inmanyrealtemporalgraphs,newnodesappear,oldnodesdisappearandreappear,and
31
edgesformbetweenbothexistingandnewlyarrivingnodes. Inthisregime,futurepredictionisno
32
longeronlyarankingproblemoverknownnodepairs;itbecomesagraphreconstructionproblem
33
withchangingsize,composition,andtopology.
34
Wearguethatthissettingismorenaturallyaddressedthroughinductivesnapshotforecasting. Givena
35
sequenceofgraphsnapshotsuptotimet,thegoalistopredictthenodesetandstructureforthenext
36
37
snapshotG(cid:98)t+1 . Thisforecast-then-predictviewalignswithdeployment: downstreamdecisionsare
madebeforethenextgraphisobserved,andmanydownstreamquantitiescanbecomputedonlyafter
38
Submittedto40thConferenceonNeuralInformationProcessingSystems(NeurIPS2026).Donotdistribute.

afuturegraphhasbeenconstructed. Thus,atemporalgraphmodelshouldnotonlyrankcandidate
39
edges,butalsoproduceacoherentfuturegraphonwhichnewtaskscanbeevaluated.
40
Topology[4]isusefulinthissettingbecauseitprovidesacompactwaytodescribegraphstructure
41
acrossscaleswhileabstractingawayfromexactnodeidentities. Twosnapshotsmaydifferinsize
42
andnodelabels,yetstillsharesimilarstructuralorganization,suchasheavy-taileddegreepatterns,
43
hub-centeredmotifs,orchangesinedgedensity. Ratherthanpredictingedgesdirectly,wesummarize
44
eachsnapshotbyamultiscalefiltrationdescriptorthatrecordshownodeandedgemassaccumulate
45
acrossfiltrationthresholds[4]. Thisdescriptorcapturesthestructuralstateofthegraph,includingits
46
finalnodeandedgebudgets,whileremaininglightweightenoughtoforecastovertime.
47
Thisleadstoourcentralformulation: temporalgraphforecastingasaninversetopologyproblem.
48
49
Insteadof predictingindividual edges, wefirstpredict thefuturetopological descriptor Φ(cid:98)(G
t+1
).
Wethenreconstructaplausiblefuturegraphwhosestructurerealizesthispredicteddescriptorwhile
50
respectinginductiveconstraintssuchasnodearrivals,nodereappearances,andedgetypesinvolving
51
new nodes. Because the descriptor is not injective, many non-isomorphic graphs may share the
52
sametopologicalsignature. Ourdecoderresolvesthisambiguitybycombiningglobaltopological
53
constraintswithlearnededge-formationscores,seekingareconstructionthathaslowdescriptorerror
54
andmatchesempiricalinteractionpatterns.
55
56
WeintroduceTOPOGED(TopologicalGraphEncoderDecoder),afullyinductiveencoder–decoder
57
frameworkfordiscrete-timetemporalgraphforecasting. TOPOGEDencodeseachsnapshotthrough
adegree-basedfiltrationdescriptorandforecaststhenextdescriptortogetherwithnodeandedge
58
budgets. Amemorymoduleestimateswhichhistoricalverticesreappear,newunlabeledverticesare
59
instantiatedaccordingtothepredictedarrivalbudget,andamulti-phasedecodergeneratesedges
60
byinductivetype: old–oldrecurring,old–oldnewlyformed,old–new,andnew–new. Theresulting
61
forecastsarecompletefuturesnapshotsandcanbeuseddirectlyfordownstreampredictionwithout
62
task-specificretraining. Ourcontributionsareasfollows:
63
• Weformulatetemporalgraphlearningundernodechurnasaninductivesnapshotforecasting
64
problem and define arrival-aware node and edge types, enabling explicit modeling of
65
reappearingoldnodes,newlyarrivingnodes,andedgesinvolvingnewnodes.
66
• Weintroduceatopology-guidedinverseformulationthatforecastsamultiscalefiltration
67
descriptorandreconstructsagraphrealizationfromit.
68
69
• WeproposeTOPOGED,atopologicalencoder–decoderthatcombinesdescriptorprediction,
memory-basednodereappearance,andmulti-phaseedgedecoding.
70
71
• WeevaluateTOPOGEDacrosssocial,webandtransactionnetworks,showingconsistent
gainsinstructuralfidelity,node-levelforecasting,edge-levelprediction,anddownstream
72
graph-propertypredictionwhilereducingthetimecostsby79%againstthebestcompetitor.
73
2 RelatedWork
74
Temporalgraphmodelslearnevolvingnodeandedgerepresentationsandthemodelsarecommonly
75
modeled either as discrete-time dynamic graphs (DTDGs) or continuous-time dynamic graphs
76
(CTDGs). CTDGsmethods[46,19,38,25,45,23,40,42,31]modelthenetworkasastreamof
77
time-stampedevents.DTDGsmodels[34,32,37,43]representthenetworkasasequenceofsnapshot
78
graphscollectedatdiscretetimestamps. Theseapproachesprovidestrongpredictors,butmostassume
79
afixednodeuniverseorpersistentnodeidentities,limitingtheirabilitytoinductivelygenerateunseen
80
nodesandtheirinteractions(SeeTable1). WefocusonDTDGsbecausesnapshot-levelmodeling
81
offersatractableframeworkforforecasting,enablingbatchprocessingovertimeandexplicitcontrol
82
ofstructuraltargets(e.g.,nodeandedgebudgetsunderchurn). Incontrast,CTDGgenerationrelies
83
onevent-levelmodelingandpoint-processassumptionsthatareunnecessaryforsnapshotforecasting.
84
Dynamicgraphgeneration. Dynamicgraphgenerationhasbeenstudiedinbothdiscrete-time[48,
85
24]andcontinuous-time[9,51,10,14]regimes. Whileeffectiveforedge-eventsynthesis,theydonot
86
directlyyieldcoherentsnapshotforecastsinthepresenceofchurnandareoftentiedtotrainingnode
87
IDs. Thisleavesagapfordiscrete-timesnapshot-forecastingmethodsthatcaninductivelypredict
88
futuregraphswithnewnodeswhilepreservingglobalstructuralfidelity.
89
2

Topologicalandspectralmethods. Acomplementarylineofworkincorporatestopologicaland
90
spectralpriorstocapturehigher-orderstructurebeyondlocalmessagepassing. SLATE[18]lever-
91
ages temporal Laplacian features but does not address snapshot generation or inductive churn.
92
Persistenthomologyandtopology-aware
93 Table1: Comparisonofrepresentativetemporalgraph
GNNshavebeenusedforgraph-levelread-
94 learningmethodsbymodelingassumptions.
outs, structural evolution, and generative
95
regularization [12, 3, 6, 13, 44, 29]. We
96 buildonthislinebyusingafiltrationsig- Method(s) n N o e d w es e N d e g w es E n d ew ge n s o w d i e t s h sna F p u s l h l ot o P b r j i e m c a ti r v y e
97
98 nature as a compact multiscale state for J D O y D SA IE T , , D E y v R ol e v p e , G T C G N A , T H , T T G G N N ✗ ✗ ✓ ✓ ✗ ✗ ✗ ✗ Nod L e i / n g k ra p p re h d p ic re ti d o i n ction
99 snapshotforecastingandgeneration. C TI T G A G N E , R H , O D P G E -Gen ✗ ✗ ✓ ✓ ✗ ✗ Par ✗ tial G E r v a e p n h t g m e o n d er e a li t n io g n
SLATE,DyGPrompt ✗ ✓ ✗ ✗ Discriminativelearning
3 Preliminaries TOPOGED(ours) ✓ ✓ ✓ ✓ Graphprediction
100
Dynamicgraphconcepts. Westudydiscrete-timedynamicgraphsrepresentedasasequenceof
101
snapshotsG ={G ,...,G }withG =(V ,E ).Weconsidertheinductivesettingwherethenodeset
102 1 T t t t
changesovertime. Foreachnodev,wedefineitsfirstappearancetimeasτ(v):=min{t:v ∈V }.
103 t
At time t, we partition the node set as V = Vnew ∪Vold, where Vnew = {v ∈ V : τ(v) = t}
104 t t t t t
andVold = {v ∈ V : τ(v) < t}. Nodesthatdisappearandlaterreappeararetreatedasoldupon
105 t t
reappearance(seeFigure1).
106
107 We decompose edges in E t into four disjoint types 2 n 1 n 0 2 0 2 n 1 0 New node
108 basedontheappearanceofedgeendpoints: old–old n n 5 n 5 6 n n Old node
109 ( o o ld o), ed o g ld es – , ne w w e ( d o is n t ) i , ng n u e i w sh –n b e e w tw ( e n e n n ). pr A ev m io o u n s g ly o o ld b – - 3 𝑮𝟎 4 3 𝑮𝟏 4 7 𝑮 n 𝟐 𝑮" 𝟑 n O Ne ld w e e d d g g e e
110
111 servedpairsandnewlyformedones.FollowingEdge- Figure1: Nodesalongwiththeirrecurringandnew
112
Bank[30],werefertotheseasoo-bank(i.e.,previ- edges.SnapshotG3willbepredicted.
ouslyseen)andoo-nobank(i.e.,notseen)edges. Formallywedefine:
113
E
t
oo−bank={(u,v)∈Et:u,v∈V
old
(t),(u,v)∈∪t
j
−
=
1
1
Ej}, E
t
on ={(u,v)∈Et:u∈V
old
(t),v∈Vnew(t)},
114
115
E
t
oo−nobank={(u,v)∈Et:u,v∈V
old
(t),(u,v)∈/∪t
j
−
=
1
1
Ej}, E
t
nn ={(u,v)∈Et:u,v∈Vnew(t)}.
Topologicalfiltration. Topologicaldataanalysis[7]introducesthenotionofafiltrationtodescribe
116
howthestructureofagraphemergesacrossscales. Bytrackingthebirthandmergingofcomponents
117
andhigher-orderfeaturesasascaleparametervaries,filtrationsprovideaprincipledmultiscaleview
118
ofgraphorganization[29,39]. However,computingfullpersistenthomologyonlargeorevolving
119
graphsisoftenprohibitivelyexpensive[1],bothcomputationallyandinmemory,whichlimitsits
120
directuseintemporalgraphlearning.Thismotivatestheuseoflightweightfiltration-basedsummaries
121
thatretainmultiscalestructuralinformationwhileremainingscalable.
122
GivenG andafiltrationfunction,suchasnodedegree,f : V → R,aninducedsublevelfiltration
123
usingthresholdscanbeformedasϵ < ··· < ϵ : V = {v ∈ V : f(v) ≤ ϵ },E = {(u,v) ∈ E :
124 1 n i i i
u,v ∈V },G ⊆···⊆G =G.Whenthefiltrationlevelisclear,weomitthesubscripti.
125 i ϵ1 ϵn
126
4 TOPOGED:TopologicalGraphEncoder-DecoderFramework
Philosophy. We propose the view that the evolution of a graph can be understood through its
127
multiscalestructuralshape,andthattopologyprovidesaprincipledwaytomakethisshapecomparable
128
across time. In temporal graphs, snapshots may differ widely in size, node identity, and local
129
connectivity due to churn, yet they often follow coherent structural patterns as they evolve. For
130
example,inblockchaintransactionnetworks[33],newnodesmusttransactwithexchangenodesto
131
buycoins,whichcreatesstar-shapedsubgraphsaroundexchangenodes.
132
Topologicalrepresentationscapturethesepatternsbysummarizinghowconnectivityaccumulates
133
acrossscales,abstractingawayfromspecificnodelabelsandexactgraphsize. Inthissense,topology
134
can provide a structural blueprint across snapshots: it preserves the graph’s organization while
135
allowingscale-andnode-specificeffectstobelearneddownstream.
136
Goalandhigh-leveldesign. Weconsiderdiscrete-timedynamicgraphswithnodechurn. Given
137
138
snapshotsuptotimet,TOPOGEDforecaststhenextsnapshotG(cid:98)t+1 =(V(cid:98)t+1 ,E(cid:98)t+1 )inaninductive
139
settingwherenewnodesmayappear. TOPOGEDconsistsofthreecomponents: (i)ascalablemulti-
scaletopologicaldescriptorΦ(G )foreachsnapshot,computedviaTDA-stylefiltrationsequences;
140 t
(ii)atopologypredictor thatforecaststhenext-snapshotdescriptorΦ(G )togetherwithcoarse
141 t+1
3

Edgebank Graph Constructor
Phase 0: Node sampling
Sample Sample edges
nodes and their edges types
Phase 1: Predict Link predictor
edges types
Graph Constructor
De
G
s
r
c
a
r
p
ip
h
t i
D
on
e s
G
c
e
ri
n
p
e
ti
r
o
a
n
tor
Probailities
Predictor
P
P
h
h
a
a
s
G
s
e
e
r a
3
2
p
:
:
h
P
P
r
r
C
e
e
o
d
d
n
i
i
c
c
s
t
t
tructor
Select top
Graph Constructor
Phase 4: Predict
Description Predictor Recompute
Graph Constructor
Figure2: TOPOGEDoverview.EachobservedsnapshotGtisencodedintoamultiscaletopologicaldescriptorΦ(Gt)via
filtration.Alightweighttopologypredictorforecaststhenext-stepdescriptorΦ(Gt+1)togetherwithcoarsestructuraltargets
(nodeandedgebudgets).Conditionedonthesetargetsandanode-memorystate,thephaseddecoderreconstructsthenext
snapshotG(cid:98)t+1.
structuraltargets,includingtheedgebudget|E |,nodebudget|V |,andtheprobabilityofnew
142 t+1 t+1
143
nodespn
t+
ew
1
;and(iii)adecoderthatreconstructsG(cid:98)t+1 bygeneratingedgesinfourstagesalignedwith
thefouredgetypesdefinedinSection3. AnoverviewofourdesignisgiveninFigure2.
144
4.1 GraphEncoder
145
ForasnapshotG =(V ,E ),theencodercomputesafiltration-basedmul-
146 t t t 𝒢 6 5
tiscaletopologicaldescriptor. Giventhedegreethresholdsϵ < ··· < ϵ ,
147 1 n
we form the sublevel filtration G ⊆ ··· ⊆ G = G . At each filtra- 2 8 1
148 t,ϵ1 t,ϵn t
tion level i, we record node and induced-edge counts, x = |V | and
149 t,i t,ϵi 4 7 3
y = |E |. ThisyieldsmonotonesequencesX = (x ,...,x )and
150 t,i t,ϵi t t,1 t,n
151 Y t = (y t,1 ,...,y t,n ). We denote the resulting topological descriptor of 𝑋 2 3 5 8
G by Φ(G ) := (X ,Y ). By definition, at the final threshold ϵ , these
152 t t t t n
𝑌 0 1 1 11
sequencesrecoverthesnapshotsizes: X =|V |andY =|E |(Fig.3).
153 t,n t t,n t
∊ ∊ ∊ ∊
! " # $
Weusethedegree-basednodefiltrationf(v)=deg(v),whichiscomputa-
154
tionallyinexpensive. Importantly,adegreeprovidesadirectandefficient Figure3: Graphencoder
155
handleontheinversetopologyproblem: predictingagraph,givenitstopo- throughdegreefiltration.
156
logicaldescription. Degreethresholdsinthedescriptorcontrolnodeinclusionandinducededge
157
counts,whichwecanusetoconstructagraphthatrealizes(i.e.,decodes)thepredictedtopological
158
descriptor.
159
ThefollowingpropositionsupportsusingΦ(G )asastableforecastingtargetbyshowingthat,for
160 t+1
afixednodeset,boundededgenoiseinducesboundeddescriptorerror;node-setchangesarehandled
161
separatelybythechurnmodel. WedefertheprooftoAppendixE.
162
Proposition4.1(Stabilityunderedgeedits). LetG = (V,E)andG′ = (V,E′)begraphsonthe
163
samenodesetwith|E△E′| ≤ δ. LetX,Y andX′,Y′ bethedegree-sublevelfiltrationnodeand
164
induced-edgecountsequencescomputedusingthresholdsϵ <···<ϵ .
165 1 n
(i) Ifthethresholdsareconsecutivedegrees{0,1,...,D},then∥X−X′∥ ≤2δ.
166 1
(ii) Foranythresholds,∥Y −Y′∥ ≤δ (cid:80)n (1+2ϵ ).
167 1 i=1 i
4.2 PredictorforanEncodedGraph
168
GiventhemostrecentkdescriptorsΦ(G ),...,Φ(G ),weusealightweighttemporalpredictor
169 t−k+1 t
170
toforecastthenext-stepdescriptorΦ(cid:98)(G
t+1
). Sincethefinalentriesofthedescriptorrecoversnapshot
171
size, this prediction also determines the node and edge budgets |V(cid:98)t+1 | and |E(cid:98)t+1 |. The snapshot
nodecountislearnedfrom|Vˆ |,butdecodingthenextsnapshotadditionallyrequiresanticipating
172 t+1
nodechurn. Wethereforejointlypredictthefractionofnewlyappearingnodesatt+1,definedas
173
174
pˆn
t+
ew
1
:= |
|
V
V
t
t
n +
+
ew 1
1|
| . Giventhepredictednodebudget|V(cid:98)t+1 |andthepredictedfractionofnewnodes
4

|     |     |     |     | (cid:4) | (cid:5) |     |     |
| --- | --- | --- | --- | ------- | ------- | --- | --- |
175 p n ew ,wedeterminethenodecountsn o ld = |V(cid:98)t+1 |(1−p n ew ) andn n ew =|V(cid:98)t+1 |−n o ld . This
(cid:98) t+ 1 (cid:98) t+ 1 (cid:98) t+ 1 (cid:98) t+ 1 (cid:98) t+ 1
separationisolatesthereappearanceproblemfromthearrivalproblem.
176
| 4.3 | MemoryModuleforNodeReappearance |     |     |     |     |     |     |
| --- | ------------------------------- | --- | --- | --- | --- | --- | --- |
177
Toidentifynold reappearingnodes,werestrictcandidatestothesetofnodesobservedinthelastk
| 178 | (cid:98)t+1 |     |     |     |     |     |     |
| --- | ----------- | --- | --- | --- | --- | --- | --- |
(cid:83)t
snapshotsmemorywindow: V = V(G ). Thiswindowlimitsvarianceandreflectsthe
| 179 |     | t−k:t | τ=t−k | τ   |     |     |     |
| --- | --- | ----- | ----- | --- | --- | --- | --- |
empiricalfactthatreappearancelikelihooddecayswithtime. Foreachnodev ∈V ,weassign:
| 180 |     |            |          |     |                            | t−k:t |     |
| --- | --- | ---------- | -------- | --- | -------------------------- | ----- | --- |
|     |     |            | (cid:18) |     | (cid:19)(cid:18) (cid:19)α |       |     |
|     |     |            |          | t−ℓ | deg                        |       |     |
|     |     | P(v,t)=exp |          | − v | v                          | fβ,   | (1) |
|     |     |            |          | λ   | deg                        | v     |     |
max
181 whereℓ isthelastobservedtimeofv,deg isitsdegreeaggregatedoverthewindow,deg is
|     | v   |     |     | v   |     |     | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
themaximumdegreeinthewindow,andf isthenumberofsnapshotsinwhichvappearswithin
| 182 |     |     | v   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
the window. The exponential term enforces recency, the normalized degree captures structural
183
prominence,andthefrequencytermcapturespersistenceacrosssnapshots. Thenormalizationby
184
185 deg keepsscoresonacomparablescaleacrossdatasets. Wekeepthesefactorsseparatetoavoid
max
186 conflatinginstantaneousconnectivitywithtemporalactivity.
Wenormalize{P(v,t)} toadistributionandsamplenold nodeswithoutreplacement. Sam-
| 187 |     | v∈Vt−k:t |     |     | (cid:98)t+1 |     |     |
| --- | --- | -------- | --- | --- | ----------- | --- | --- |
plingintroducesdiversityandpreventsconcentrationonasmallsetofhigh-degreenodes,which
188
189 improves stability under heavy-tailed degree distributions. We then introduce |V(cid:98) n ew | unlabeled
t + 1
190 nodestocompleteV(cid:98)t+1 . Thissteptreatsarrivalsasacountpredictionproblemandavoidsimposing
191 identitieswherenoneexist.
| 4.4 | Filtration-GuidedEdgeBudgets |     |     |     |     |     |     |
| --- | ---------------------------- | --- | --- | --- | --- | --- | --- |
192
193 TOPOGEDreconstructsthepredictedsnapshotbygeneratingedgesinfourstagesalignedwiththe
194 edgetypesinSection3: old–oldbank(Eoo-bank), old–oldnobank(Eoo-nobank), old–new(Eon), and
new–new(Enn). Decodingisdrivenbythreesourcesofinformation: (i)nodechurntargetsproduced
195
bythepredictorandmemorymodule,(ii)edge-typebudgets,and(iii)optionalre-encodingbetween
196
stagestoupdatenoderepresentationsasstructure(throughedges)accumulates.
197
Edge-type budgets. Let |E(cid:98)t+1 | be the predicted next snapshot edge count. We predict
198
the edge-type proportions from their recent empirical history. For each snapshot, we com-
199
|          | (πoo−bank,πoo−nobank,πon,πnn), |     |     |       | πz |Ez|/|E |                   |          |
| -------- | ------------------------------ | --- | --- | ----- | ---------- | ----------------- | -------- |
| 200 pute | π t =                          |     |     | where | =          | t | for each edge | type z ∈ |
|          | t                              | t   | t t |       | t t        |                   |          |
201 {oo-bank,oo-nobank,on,nn}. Giventhelastkproportionvectors,weforecastπ withthesame
(cid:98)t+1
lightweighttemporalpredictorusedforthedescriptorvariablesandrenormalizeittolieinthesimplex.
202
|     |     |     |     |     |     | z   | z   |
| --- | --- | --- | --- | --- | --- | --- | --- |
203 Thepredictedtotaledgebudget|E(cid:98)t+1 |isthendividedacrossedgetypesascˆ =⌊π (cid:98)t |E(cid:98)t+1 |⌋.
|     |     |     |                                                      |     |     | t +1 | +1                  |
| --- | --- | --- | ---------------------------------------------------- | --- | --- | ---- | ------------------- |
|     |     |     | Inadditiontothefinalnodeandedgebudgets(|V(cid:98)t+1 |     |     |      | |,|E(cid:98)t+1 |), |
204 Filtration-guidedbudgetguidance.
205 the filtration sequences Φ(G t ) := (X t ,Y t ). provide multiscale structural targets. We use these
sequencestodefineintermediatecheckpointsoncumulativenodeandedgecountsaftereachdecoding
206
phase,andapplyasoftpenaltywhenthepartiallygeneratedgraphdeviatesfromthepredictedfiltration
207
profile. Thisguidancesteersgenerationtowardtheforecastedmultiscalestructurewithoutenforcing
208
harddegreeconstraintsatindividualfiltrationlevels. Weelaborateonhowthetopologicaldescriptor
209
| 210 Φ(G | t )guidegraphconstructionstepbystepinAppendixD.1. |     |     |     |     |     |     |
| ------- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- |
Phase1: old–oldbankedges. Webeginbygeneratingre-occurringedgesbetweenoldnodes. The
211
candidatesetconsistsofpreviouslyseenedgeswhoseendpointsareoldnodes,{(u,v) ∈ E :
| 212 |     |     |     |     |     |     | t−i:t |
| --- | --- | --- | --- | --- | --- | --- | ----- |
u,v ∈Vold }. Thesepairsarescoredbyadedicatedpredictorhead,andthehighest-scoringpairsare
| 213 | t+1 |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
214 addeduntilreachingcˆoo-bank.
t+1
Phase2: old–oldnobankedges. Next,wegeneratepreviouslyunseenedgesbetweenoldnodes.
215
Candidatepairs(u,v)∈Vold ×Vold arescoredbyadedicatedpredictorhead,andthehighest-scoring
| 216 |     | t+1 t+1 |     |     |     |     |     |
| --- | --- | ------- | --- | --- | --- | --- | --- |
217 pairsareaddeduntilreachingcˆoo-nobank.
t+1
Phase3:old–newedges.Wethenconnectnewnodestotheexistinggraph. Newnodesareinitialized
218
withcold-startembeddings(Section4.5),afterwhichold–newcandidatepairsarescoredandselected
219
| untilreachingcˆon | .   |     |     |     |     |     |     |
| ----------------- | --- | --- | --- | --- | --- | --- | --- |
| 220               | t+1 |     |     |     |     |     |     |
5

221 Phase 4: new–new edges. Finally, we generate edges among new nodes by scoring new–new
candidatepairsandaddingedgesuntilthebudgetcˆnn ismet. Thisproducesthecompletedsnapshot
| 222                |                                                        |     |                     |     | t+1 |     |     |     |     |
| ------------------ | ------------------------------------------------------ | --- | ------------------- | --- | --- | --- | --- | --- | --- |
| 223 G(cid:98)t+1 . | Aftergeneration,weupdatethenodememoryusingG(cid:98)t+1 |     |                     |     |     |     | .   |     |     |
| 4.5 EdgeDecoder:   |                                                        |     | Multi-HeadPredictor |     |     |     |     |     |     |
224
Edgegenerationisperformedbyamulti-headpredictorwithasharedembeddingtrunk. Ateach
225
decodingphase,thetrunkcomputesnodeembeddings{h } onthecurrentpartiallyconstructed
| 226 |     |     |     |     |     | v   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
v∈V(cid:98)t+1
graphusingaGCN[21]overthegeneratedadjacency. Fourtype-specificheadsthenscorecandidate
227
pairscorrespondingtotheinductiveedgecategories:
228
|     |     |         |            | )(cid:55)→p(oo-bank), |     |           |            | )(cid:55)→p(oo-nobank), |     |
| --- | --- | ------- | ---------- | --------------------- | --- | --------- | ---------- | ----------------------- | --- |
|     | M   | oo-bank | :(h u ,h v |                       | M   | oo-nobank | :(h u ,h v |                         |     |
|     |     |         |            | uv                    |     |           |            | uv                      |     |
|     |     | M       | :(h ,h     | )(cid:55)→p(on),      |     | M         | :(h ,h     | )(cid:55)→p(nn).        |     |
|     |     | on      | u v        | uv                    |     | nn        | u v        | uv                      |     |
Foredgetypez,thecorrespondingheadoutputsprobabilitiespˆz ∈[0,1]forcandidatepairs(u,v).
| 229 |     |     |     |     |     |     | uv  |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Candidatepairsarerankedbypˆz
230 uv ,andthetop-rankededgesareaddeduntilthephase-specificbudget
cz
231 (cid:98)t+1 isreached. Tokeepdecodingtractable,eachphasescoresonlyasampledcandidatepool,whose
sizeischosenproportionaltothepredictededgebudget.
232
Cold start for new nodes. New nodes introduced in phases 3 and 4 initially have no incident
233
edgesandthereforelackstructuralcontext. Weinitializetheirembeddingsusingdegree-conditioned
234
235 prototypescomputedfromthepredictedoldnodes. Letκ(v)denotethedegreebinassignedtoanew
236 nodevfromthepredictedfiltrationandedge-budgettargets.
1 (cid:80)
Forbink,define C ={u∈V(cid:98) o ld :deg(u)=k},andhn ew = h .WhenC
| 237 |     |     | k   | t + 1 |     | v   | |C  | | u∈Cκ(v) | u κ(v) |
| --- | --- | --- | --- | ----- | --- | --- | --- | --------- | ------ |
κ(v)
238 isempty,weusethenearestnon-emptybin,andfallbacktoaglobalprototypeifnosuchbinexists.
ThebinningschemeisdetailedinAppendixD.3.
239
Since the filtration descriptor is not injective, many non-isomorphic graphs can realize the same
240
predicteddescriptor. Themulti-phasedecoderresolvesthisambiguitybycombiningglobalfiltration-
241
242 guidedtargetswithlearnededgescorestrainedfromhistoricalpositiveandnegativeedges.Thisyields
243 graphrealizationsthatareencouragedtomatchthepredictedmultiscalestructurewhileremaining
consistentwithempiricaledge-formationpatterns.
244
| 245 4.6 TrainingandInference |     |     |     |     |     |     |     |     |     |
| ---------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
246 TOPOGEDistrainedbysupervisingboththeevolutionofthetopologicaldescriptorandtheedge
generation,usingtwocorrespondingcomponents. First,thetopologypredictorforecaststhenext-
247
stepdescriptorandprobabilities. Giventhepastkencodings,itoutputsΦ(cid:98)(G )andthepredicted
| 248 |     |     |     |     |     |     |     | t+1 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
new-nodefractionpnew,andisoptimizedwitharegressionlossagainstthetruedescriptorsequence.
| 249 |     | (cid:98)t+1 |     |     |     |     |     |     |     |
| --- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
Second, the edge decoder is trained with positive and negative edges from historical snapshots.
250
For each inductive edge type z ∈ {oo-bank,oo-nobank,on,nn}, let Ez be the corresponding
| 251 |     |     |     |     |     |     |     | t+1 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ground-truthedgesinsnapshott+1. Eachdecoderheadistrainedusingbinarycross-entropywith
252
negativesampling:Lz =− (cid:80) logpz − (cid:80) log (cid:0) 1−pz (cid:1) ,whereNzdenotes
| 253 |     |     | edge | (u,v)∈Ez | uv  | (u′,v′)∼Nz |     | u′v′ |     |
| --- | --- | --- | ---- | -------- | --- | ---------- | --- | ---- | --- |
t+1
uniformlysamplednegativepairsfromthecorrespondingcandidatespace. Thisencouragesdecoded
254
graphstomatchempiricaledge-formationpatterns. Duringtraining,weuseteacherforcing[41]:
255
256 eachdecodingphaseconditionsontheground-truthpartialgraphformedbyearlier-phaseedgesin
(cid:80)
257 G ,stabilizingoptimizationanddecouplingphase-wiseerrors. Thetotaldecoderlossis Lz ,
| t+1 |     |     |     |     |     |     |     |     | z edge |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ |
andthetopologypredictoranddecoderaretrainedwithseparateoptimizers.
258
Inference. Atinferencetime,givensnapshotsuptotimetandthecurrentmemorystate,wepredict
259
Φ(cid:98)(G ),thenodeandedgebudgets(|V(cid:98)t+1 |,|E(cid:98)t+1 |),andthenew-nodefractionp n ew . Wethenselect
| 260 t+1 |     |     |     |     |     |     |     | (cid:98) | t+ 1 |
| ------- | --- | --- | --- | --- | --- | --- | --- | -------- | ---- |
261 reappearingnodeswiththememorymodule,instantiatenewnodes,andreconstructG(cid:98)t+1 byrolling
outthefourdecodingphasesautoregressively.
262
Computational costs. The dominant per-snapshot computational cost of TOPOGED is
263
| (cid:16) |     |     |     | (cid:17) |     |     |     |     |     |
| -------- | --- | --- | --- | -------- | --- | --- | --- | --- | --- |
O M|V(cid:98)t+1 |s(d)+L|E(cid:98)t+1 |d , where M is the number of sampled candidate partners per node,
264
s(d)isthecostofscoringoneedgeatembeddingdimensiond,andListhenumberofGCNlayers;
265
thusedgescoringandmessagepassingdominateruntime. WedetailthecostbreakdowninApp.C.4.
266
6

5 Experiments
267
268
Givenasequenceofsnapshots{G
t
}T
t=1
,weforecastthenextsnapshotG(cid:98)t+1 usinginformationupto
timet,andthenevaluatedownstreamperformancewithaforecast-then-predictpipeline.
269
Table2: Summaryofdatasetstatistics.
270
Datasets. We evaluate TOPOGED
on 14 temporal interaction bench-
2 2 7 7 1 2 marks: CollegeMessage[26],Math- Dataset G N ra u p m hs A |V vg | A |E vg | A V vg old % o A o v - g ba % nk oo A - v n g ob % ank Av o g n % Av n g n %
273 Overflow[27]andReddit-Body[22], CollegeMsg 181 123 185 91 49 39 10 3
MathOverflow 189 142 142 82 28 43 22 6
TGBL-Wiki[15]and10ERC20token
274 Adex 299 249 200 51 28 10 52 9
transaction networks extracted from Aeternity 235 303 282 54 34 12 43 11
275 Aion 196 285 239 47 24 13 49 14
276 the Ethereum blockchain [34]. The Aragon 343 336 309 45 18 4 74 4
Bancor 317 275 224 47 24 9 54 13
277 ERC20 datasets exhibit substantial Centra 264 264 252 49 23 14 55 8
nodechurn, providingachallenging Cindicator 226 300 264 48 24 13 47 16
278 Coindash 274 282 267 58 31 14 52 3
279 testbedforinductiveforecasting. As DGD 725 69 61 61 38 10 47 5
Iconomi 548 179 158 55 34 8 54 5
280 we show in App.G, edge-type distri- Reddit_B 405 403 272 87 43 38 17 2
butionsshiftmarkedlyovertime. Ta- TGBL-Wiki 31 1451 1284 78 53 21 20 6
281
ble2summarizesdatasetstatistics. Note:Avg %metricsrepresentindependentaveragesacrossallsnapshots;totalsmaynotsumto
282
100%duetooutliervarianceandtemporalskew.
283
Temporalpredictors. ForpredictingΦ(G(cid:98)t+1 )andprobabilities,weevaluatedSMA[47],VAR[36],
V-EWMA [2], SSM [17], and VECM [16]. V-EWMA was adopted as it demonstrated 0.50%
284
improvementoverthenearestcompetitorinnodecountpredictionand0.88%improvementoverthe
285
nearestcompetitorinedgecountprediction(SeeApp.C.3).
286
287
Edgepredictionmethods. WecompareTOPOGEDagainstprominentandrecenttemporalGNNs:
ROLAND[45],EvolveGCN[28],VGRNN[11],GC-LSTM[5],andHTGN[43].Furtherinformation
288
aboutthesemodelsisprovidedinApp.C.1. Modelsaretrainedfor25to100epochs, withearly
289
stoppingappliedbasedonvalidationlosswithapatienceof10epochs. BinaryCross-Entropylossis
290
usedforthemulti-headlinkpredictiontask. OptimizationisperformedusingAdam[20]. Eachmodel
291
undergoeshyperparametergridsearch,withthebestparametersbeingselectedbasedonvalidation
292
AUCover3independentruns. ModificationstothetemporalGNNs,includingparametertuning,are
293
discussedinAppendixD.5. Eachdatasetispartitionedinto70%fortraining,15%forvalidation,and
294
15%fortesting.
295
ComputationalEnvironment. ExperimentswereconductedonaclusterwithNVIDIAH200NVL
296
GPU(141GB)nodes,dual64-coreAMDEPYC9555processors(128physicalcores),and1.5TB
297
ofsystemRAM.
298
Reproducibility. Ourcodeisavailableathttps://anonymous.4open.science/r/TopoGED.
299
Hyperparameters. AllneuralmodelsaretrainedwithAdamusinganinitiallearningrateof10−3
300
301
andearlystoppingonvalidationperformancewithpatience10. ForTOPOGED,thedescriptorand
302
probabilitypredictoraretrainedwithregressionlossonΦ(cid:98)(G
t+1
),p
(cid:98)
n
t+
ew
1
,andπ
(cid:98)t+1
. Theedgedecoder
modelselectionusesvalidationROC-AUC.Allmodelsaretunedonavalidationwindowunderthe
303
304
sameinformationbudget,withhyperparametersselectedbygridsearch. TOPOGED useshistory
windowk =5,batchsize32,hiddendimension128,oneGCNlayer,dropout0.1,andweightdecay
305
0. Weevaluateeachtaskover3independentruns.
306
Metrics. Wereportresultsinfourstages. First,weevaluatedescriptorpredictionbyforecastingthe
307
terminalnodeandedgecountsencodedinΦ(G )andcomparesixtemporalforecastingmethodsin
308 t+1
AppendixC.3. Second,weevaluateconstructedgraphsatthestructurelevelusingtwelvemetrics,
309
including average degree, triangle count, density, clustering coefficient, and descriptor ℓ error.
310 2
Third,weevaluatenode-levelbehaviorusingeightmetricsthatmeasureold-nodereappearanceand
311
new-nodecountaccuracy. Finally,weevaluateedge-levelbehaviorusingelevenmetricsthatmeasure
312
edge formation across inductive types, including precision and recall for old–old edges. Metric
313
definitionsaregiveninSectionC.2.
314
Graph prediction. Table 3 reports wins and aggregate scores across structure, node, and edge
315
316
evaluations. TOPOGEDdominatesstructuremetricswithamacroaverageof7.21winscompared
to2.64forthestrongestbaseline. Thisadvantageisconsistentacrossdatasets,withnear-complete
317
coverageonCentraandDGD.Theseresultsshowthatthepredicteddescriptortranslatesintoaccurate
318
globalpropertiesintheconstructedgraphs.
319
7

Table3:Forecastresults.Foreachdatasetandmetricgroup,wereportTOPOGEDandthestrongestbaseline.
Structuremetricwins↑ Nodemetrics↑ Edgemetrics↑
Dataset
TOPOGED Bestbaseline TOPOGED Bestbaseline TOPOGED Bestbaseline
CollegeMsg 8 4(EvolveGCN) 0.44±0.10 0.04±0.02(EvolveGCN) 0.19±0.07 0.00±0.00(multiple)
MathOverflow 9 2(multiple) 0.41±0.04 0.01±0.01(multiple) 0.02±0.01 0.00±0.00(multiple)
Adex 6 4(HTGN) 0.54±0.09 0.09±0.02(GCLSTM) 0.09±0.04 0.01±0.00(GCLSTM)
Aeternity 9 2(multiple) 0.46±0.11 0.01±0.01(ROLAND) 0.11±0.04 0.00±0.00(multiple)
Aion 6 3(multiple) 0.45±0.05 0.01±0.00(GCLSTM) 0.15±0.05 0.00±0.00(multiple)
Aragon 6 4(HTGN) 0.51±0.04 0.01±0.00(GCLSTM) 0.22±0.08 0.00±0.00(multiple)
Bancor 6 4(HTGN) 0.57±0.07 0.03±0.01(GCLSTM) 0.14±0.07 0.00±0.00(multiple)
Centra 5 4(ROLAND) 0.47±0.14 0.01±0.01(multiple) 0.07±0.06 0.00±0.00(multiple)
Cindicator 6 4(ROLAND) 0.51±0.04 0.02±0.01(GCLSTM) 0.13±0.07 0.00±0.00(multiple)
Coindash 5 4(HTGN) 0.47±0.12 0.03±0.01(GCLSTM) 0.13±0.06 0.00±0.00(multiple)
DGD 8 3(EvolveGCN) 0.53±0.09 0.02±0.01(GCLSTM) 0.08±0.05 0.00±0.00(multiple)
Iconomi 6 3(multiple) 0.51±0.09 0.01±0.01(ROLAND) 0.10±0.06 0.00±0.00(multiple)
Reddit-B 8 2(multiple) 0.38±0.03 0.00±0.00(multiple) 0.03±0.02 0.00±0.00(multiple)
TGBL-Wiki 7 4(multiple) 0.65±0.01 0.35±0.01(HTGN) 0.18±0.01 0.01±0.00(HTGN)
Macroavg. 6.78±1.32 3.35±0.81(HTGN) 0.49±0.07 0.05±0.09(GCLSTM) 0.12±0.06 0.00±0.01(multiple)
Structuremetricwinsisthenumberoftimesthateachmodelhasbeenclosestto0percenterroroverassortativitycoefficient,clusteringcoefficient,
averagedegree,density,trianglecount,extraedges,extranodes,missingedges,andmissingnodes.NodemetricsdisplaytheF1ofallnodes.
EdgemetricsaretheF1scoresacrossalledgetypes.DetailedresultscanbeseeninAppendixFandC.2.
320
Atthenodelevel,TOPOGEDachievesamacroaveragescoreof0.49,whilethestrongestbaseline
reaches0.05. ThegapisstableacrossdatasetsandislargestonBancorandDGD.Baselinenode
321
performance is limited by the absence of explicit node-set prediction. Standard temporal graph
322
modelsscoreedgesoverafixedorsamplednodeuniverseandrelyonnegativesamplingforAUC
323
evaluation. Thissetupmeasuresrankingqualitywithinacandidatesetbutdoesnotconstrainthesize
324
orcompositionofVˆ . Whenthesemodelsareusedforfullgraphconstruction,theactivenode
325 t+1
setisinducedimplicitlybypredictededgesandcangrowwithoutcontrol. Thisexplodingnode-set
326
327
problemleadstolowprecisionandrecallfornodereappearance. Incontrast,TOPOGEDpredicts
nodebudgets, selectsreappearingnodesthroughthememorymodule, andintroducesnewnodes
328
explicitly,whichyieldsconsistentimprovements.
329
330
Table4: AblationforCollegeMsg At the edge level, TOPOGED at-
tainsamacroaverageof0.12com-
331
332
Metric F.Probs F.Φ(G(cid:98)t) Oracle TOPOGED paredto0.01forthestrongestbase-
AvgNodeDegree 3.63±4.91 0.17±0.35 0.29±0.09 0.36±0.19 line. The improvement appears
333 UniqueDegreeCount 2.32±0.82 0.90±0.55 0.60±0.32 0.64±0.48
334 DegreeCentrality 0.02±0.82 0.09±0.35 0.29±0.09 0.28±0.24 acrossalldatasets,withcleargains
AssortativityCoefficient −0.87±14.05 2.53±5.82 1.59±6.98 1.13±7.92
335 ClusteringCoefficient 0.75±2.82 0.34±1.25 0.05±0.05 0.07±0.06 onAragon,Coindash,andTGBL-
Density 0.02±0.82 0.09±0.35 0.29±0.09 0.28±0.24 Wiki. Baselines often achieve
336 NumTriangles 7929.25±31327.34 7.32±6.29 1.86±2.16 3.11±3.20
337 DescriptorNorm 906.68±1901.83 41.74±10.11 27.67±7.88 35.42±11.91 near-zero scores because they do
MedianExtraNodes 123.00±12.85 0.00±1.15 0.00±0.00 0.00±1.16 not model inductive edge types
338 MedianMissingNodes 0.00±0.00 0.50±1.18 0.00±0.00 0.50±1.18
339 MedianExtraEdges 413.50±47.61 6.00±2.80 8.00±1.25 7.50±2.67 andthereforefailtoallocateedges
MedianMissingEdges 0.00±0.00 0.00±0.12 0.00±0.00 0.00±0.00
acrossold–old,old–new,andnew–
340
Forallmetricscloserto0isbetter. Boldindicatesbest,underlinedissecond-best.
new categories. Methods suchas
341
VGRNNmayappearcompetitiveonisolatedmetricsbyoverpredictingedges,whichincreasesrecall
342
onsampledevaluationsbutintroducesstructuralnoiseinfull-graphreconstruction. HTGNperforms
343
competitivelyondatasetswithdominantold–oldinteractionssuchasCollegeMsg,wheremostedges
344
fallintothiscategory. Thismasksitslimitationtoold–oldedges. However,itdoesnotmodelnode
345
churnorinductiveedgetypesanddegradesundersettingswithhigherturnover.
346
0.19
0.18
0.17
0.16
0.15
0.14
0.13
0 5 10 15 20 25 30 k
1F
egdE
347
TOPOGED predicts node counts, edge counts, and all inductive
edgetypesexplicitly. Italignsnodeselection,edgeallocation,and
348
globalstructurewiththepredicteddescriptor. Thisalignmentyields
349
consistent gains across structure, node, and edge evaluations and
350
producesforecastedgraphsthatremaincoherentundernodechurn
351
andstructuraldrift.
352
Ablations. We run three ablations. In False Φ, we replace the
353
354 descriptorΦ(cid:98)(G t+1 )witharandommonotonesequence(X(cid:101),Y(cid:101))that
preserves the final node and edge counts but discards multiscale
355
356
structure. InFalseProbs,weretainΦ(cid:98)(G
t+1
)butreplacenodeand
Figure 4: Sensitivity of edge
edge-typeprobabilitieswithrandomvalues. InOracle,weprovide
357 prediction to the history win-
thedecoderwiththeground-truthdescriptorandprobabilitiesfrom
358 dowkinTGBL-Wiki.
t+1,whichservesasanupperboundforreconstruction.
359
8

360 Table 4 shows that each component controls a distinct failure mode. Randomizing probabilities
(F.Probs)leadstosevereover-generation,withlargeerrorsintrianglecountanddescriptornorm,
361
indicatingthatcorrectedgeallocationrequirescalibratednodeandedge-typeprobabilities.Incontrast,
362
corrupting the descriptor (F. Φ) preserves budgets but distorts higher-order structure, producing
363
364 large deviations in assort. and clustering coefficient. This confirms that matching final counts
365 alone is insufficient; the multiscale profile in Φ is necessary to recover structural organization.
| The oracle | configuration |     |                                        |     |     |     |
| ---------- | ------------- | --- | -------------------------------------- | --- | --- | --- |
| 366        |               |     | Table5: Resourceusageacross14datasets. |     |     |     |
| yields the | lowest errors |     |                                        |     |     |     |
367
| across nearly | all metrics, |       |             |               |     |              |
| ------------- | ------------ | ----- | ----------- | ------------- | --- | ------------ |
| 368           |              | Model | Time↓(sec)  | RAM↓(MB)      |     | GPU↓(MB)     |
| as expected.  | More im-     |       |             |               |     |              |
| 369           |              |       | 319.3±235.7 | 4512.7±3315.6 |     | 1124.5±569.4 |
ROLAND
370 portantly, TOPOGED ap- TGCN 28.1±19.1 6251.4±3829.5 869.8±528.4
371 proachesthisoraclewithout GCLSTM 686.7±365.2 6183.0±3805.5 1534.1±877.1
access to ground-truth fu- VGRNN 432.3±269.5 3135.9±1093.7 35359.2±22840.8
372
ture information. The gap HTGN 391.2±377.8 5950.0±3578.7 2923.0±1646.9
373
between TOPOGED and E v o l ve G C N 9 2 .0 ± 4 9 . 6 6 4 11 . 0 ± 3 8 5 3 . 3 1 35 1. 7± 8 7 8 . 8
374
|                |                 | T O P O G E | D (ours) 1 9 .0 | ± 1 2 . 2 2 1 9 6 . 4 ± | 3 5 1 . 5 | 83 9 .0 ± 14 6 9 . 6 |
| -------------- | --------------- | ----------- | --------------- | ----------------------- | --------- | -------------------- |
| 375 the oracle | is small across |             |                 |                         |           |                      |
| 376 structure  | metrics, while  |             |                 |                         |           |                      |
both ablations degrade substantially. This shows that accurate reconstruction requires the joint
377
useofdescriptorpredictionandprobabilisticdecoding,andthatTOPOGEDeffectivelyalignsthese
378
componentstorecoverbothglobalstructureandnode-levelbehavior.
379
380 Efficiency. AsTable5summarizes,TOPOGEDisscalableandusessubstantiallyfewercomputa-
381 tionalresourcesthanthebaselines.
|                                         |                               |     |        | Interaction:  |  and  |               |
| --------------------------------------- | ----------------------------- | --- | ------ | ------------- | ----- | ------------- |
| Sensitivity.                            | Figures5and4showthatthememory |     |        |               |       |               |
| 382                                     |                               |     |        | 14            |       | Optimal 0.555 |
| moduleisstableoverabroadparameterrange. |                               |     | Theα–β |               |       |               |
383
| sweepshowsalargehigh-F1regionformoderateαand |     |     |     | 12  |     | 0.510 |
| -------------------------------------------- | --- | --- | --- | --- | --- | ----- |
384
385 high β, indicating that persistence across recent snap- 10 0.465
erocS 1F tseT
386 shotsisthestrongestsignalfornodereappearance,while
|     |     |     |     | 8   |     | 0.420 |
| --- | --- | --- | --- | --- | --- | ----- |
degreeshouldreceiveanonzerobutnotdominantweight.
387
| Very large | α values reduce | performance, | since they |     |     | 0.375 |
| ---------- | --------------- | ------------ | ---------- | --- | --- | ----- |
| 388        |                 |              |            | 6   |     |       |
overemphasizehigh-degreenodesandweakenthecon-
| 389                            |     |                   |     |     |     | 0.330 |
| ------------------------------ | --- | ----------------- | --- | --- | --- | ----- |
| tributionoftemporalrecurrence. |     | Thehistory-window |     | 4   |     |       |
390
| 391 sweepshowsthatperformanceimprovessharplyfrom |     |     |     | 2   |     | 0.285 |
| ------------------------------------------------ | --- | --- | --- | --- | --- | ----- |
392 veryshortmemorywindowsandthensaturatesaround
|     |     |     |     | 0   |     | 0.240 |
| --- | --- | --- | --- | --- | --- | ----- |
intermediate values of k. Longer windows add little 0 2 4 6 8 10
393
benefitandcanintroducestalecandidates,supporting
394
theuseofafiniterecentmemoryforreappearing-node
| 395 |     |     |     | Figure5: Sensitivityofnodereappearance |     |     |
| --- | --- | --- | --- | -------------------------------------- | --- | --- |
396 selection. WeprovidefurtheranalysisaboutTOPOGED performancetoαandβ inTGBL-Wiki.
397 nodepredictorinAppendixD.2.
Limitations. TOPOGEDforecastsdiscrete-timesnapshotsratherthanevent-levelinteractionstreams.
398
Newlyarrivingnodesaretreatedasunlabeleduntiltheyreceiveedges,sothemodelpredictsthe
399
numberandstructuralrolesofnewnodes, nottheirexternalidentitiesbeforeobservation. These
400
401 assumptionsmatchtheinductivesnapshot-forecastingsettingstudiedhere,whilecontinuous-time
402 eventgenerationandidentitypredictionforunseennodesremainnaturaldirectionsforfuturework.
6 Conclusion
403
WeintroducedTOPOGED,atopology-guidedencoder–decoderframeworkforinductivetemporal
404
graphforecastingundernodechurn. Ratherthanreducingfuturepredictiontoedgerankingovera
405
fixednodeset,TOPOGEDforecastsamultiscalefiltrationdescriptor,predictsnodeandedgebudgets,
406
407 anddecodesacompletefuturesnapshotthrougharrival-awareedgegeneration. Thisformulation
408 makes node reappearance, new-node arrival, edge-type allocation, and global structural fidelity
explicitpartsofthepredictionproblem. Acrossdiversetemporalgraphbenchmarks, TOPOGED
409
producesforecastedgraphsthatbettermatchfuturenodesets,edgestructure,andgraph-levelstatistics
410
thantheexistingtemporalgraphmodels,whileremainingcomputationallyefficient. Theseresults
411
412 suggestthatfiltration-guidedsnapshotreconstructionisaviablealternativetopurelydiscriminative
413 temporalgraphprediction,andprovideasteptowardtemporalgraphmodelsthatcanforecastcoherent
414 futurenetworkstatesratherthanonlyscoreisolatedfutureinteractions.
9

References
415
[1] CuneytGAkcora,MuratKantarcioglu,YuliaGel,andBarisCoskunuzer. Reductionalgorithms
416
forpersistencediagramsofnetworks: Coraltdaandprunit. InAdvancesinNeuralInformation
417
ProcessingSystems,volume35,pages25046–25059,2022.
418
[2] RobertGoodellBrown. Statisticalforecastingforinventorycontrol. McGraw-Hill,1959.
419
[3] MathieuCarrière,FrédéricChazal,YuichiIke,ThéoLacombe,MartinRoyer,andYuheiUmeda.
420
Perslay: Aneuralnetworklayerforpersistencediagramsandnewgraphtopologicalsignatures.
421
InAISTATS,ProceedingsofMachineLearningResearch,2020.
422
[4] FrédéricChazalandBertrandMichel. Anintroductiontotopologicaldataanalysis:fundamental
423
andpracticalaspectsfordatascientists. FrontiersinArtificialIntelligence,4:108,2021.
424
[5] Jinyin Chen, Xueke Wang, and Xuanheng Xu. GC-LSTM: graph convolution embedded
425
LSTM for dynamic network link prediction. Appl. Intell., 52(7):7513–7528, 2022. doi:
426
10.1007/S10489-021-02518-9. URLhttps://doi.org/10.1007/s10489-021-02518-9.
427
[6] Yuzhou Chen, Baris Coskunuzer, and Yulia Gel. Topological relational learning on graphs.
428
AdvanceinNeuralInformationProcessingSystems,34:27029–27042,2021.
429
[7] TamalKrishnaDeyandYusuWang. Computationaltopologyfordataanalysis. Cambridge
430
UniversityPress,2022.
431
[8] ZhengZhao Feng, Rui Wang, TianXing Wang, Mingli Song, Sai Wu, and Shuibing He. A
432
comprehensivesurveyofdynamicgraphneuralnetworks: Models,frameworks,benchmarks,
433
experimentsandchallenges. IEEETransactionsonKnowledgeandDataEngineering,2025.
434
[9] AlessioGravina,GiulioLovisotto,ClaudioGallicchio,DavideBacciu,andClaasGrohnfeldt.
435
Longrangepropagationoncontinuous-timedynamicgraphs. InInternationalConferenceon
436
MachineLearning,pages16206–16225.PMLR,2024.
437
[10] Shubham Gupta, Sahil Manchanda, Srikanta Bedathur, and Sayan Ranu. Tigger: Scalable
438
generativemodellingfortemporalinteractiongraphs. InProceedingsoftheAAAIConference
439
onArtificialIntelligence,volume36,pages6819–6828,2022.
440
[11] Ehsan Hajiramezanali, Arman Hasanzadeh, Nick Duffield, Krishna Narayanan, Mingyuan
441
Zhou, and Xiaoning Qian. Variational graph recurrent neural networks. arXiv preprint
442
arXiv:1908.09710,2019. RevisedApr2020.
443
[12] ChristophHofer,FlorianGraf,BastianRieck,MarcNiethammer,andRolandKwitt. Graph
444
filtrationlearning. InInternationalConferenceonMachineLearning,pages4314–4323,2020.
445
[13] MaxHorn, EdwardDeBrouwer, MichaelMoor, YvesMoreau, BastianRieck, andKarsten
446
Borgwardt. Topological graph neural networks. In International Conference on Learning
447
Representations,2021.
448
[14] RyienHosseini,FilippoSimini,VenkatramVishwanath,andHenryHoffmann. Adeepproba-
449
bilisticframeworkforcontinuoustimedynamicgraphgeneration. InProceedingsoftheAAAI
450
ConferenceonArtificialIntelligence,volume39,pages17249–17257,2025.
451
[15] ShenyangHuang,FarimahPoursafaei,JacobDanovitch,AndrePinheiro,GuillaumeRabusseau,
452
SaharReutskaya,GuillaumeRabusseau,andReihanehRabbany. Tgb: Alarge-scalebenchmark
453
forlearningontemporalgraphs.InThirty-seventhConferenceonNeuralInformationProcessing
454
Systems(NeurIPS)DatasetsandBenchmarksTrack,2023.
455
[16] SørenJohansen. Estimationandhypothesistestingofcointegrationvectorsingaussianvector
456
autoregressivemodels. Econometrica: JournaloftheEconometricSociety,pages1551–1580,
457
1991.
458
[17] RudolfEmilKalman. Anewapproachtolinearfilteringandpredictionproblems. Journalof
459
BasicEngineering,82(1):35–45,1960.
460
10

[18] YannisKarmim,MarcLafon,RaphaëlFournier-S’Niehotta,andNicolasThome.Supra-laplacian
461
encoding for transformer on dynamic graphs. Advances in Neural Information Processing
462
Systems,37:17215–17246,2024.
463
[19] SeyedMehranKazemi. Dynamicgraphneuralnetworks. InGraphneuralnetworks: Founda-
464
tions,frontiers,andapplications,pages323–349.Springer,2022.
465
[20] DiederikPKingmaandJimmyLeiBa. Adam: Amethodforstochasticgradientdescent. In
466
ICLR:internationalconferenceonlearningrepresentations,pages1–15,2015.
467
[21] ThomasN.KipfandMaxWelling. Semi-supervisedclassificationwithgraphconvolutional
468
networks. InInternationalConferenceonLearningRepresentations(ICLR),2017.
469
[22] SrijanKumar,WilliamLHamilton,JureLeskovec,andDanJurafsky. Communityinteraction
470
andconflictontheweb. InProceedingsofthe2018WorldWideWebConferenceonWorldWide
471
Web,pages933–943.InternationalWorldWideWebConferencesSteeringCommittee,2018.
472
[23] Srijan Kumar, Xikun Zhang, and Jure Leskovec. Predicting dynamic embedding trajectory
473
in temporal interaction networks. In Proceedings of the 25th ACM SIGKDD International
474
ConferenceonKnowledgeDiscoveryandDataMining(KDD),2019.
475
[24] FanLi,XiaoyangWang,DaweiCheng,CongChen,YingZhang,andXueminLin. Efficient
476
dynamicattributedgraphgeneration. In2025IEEE41stInternationalConferenceonData
477
Engineering(ICDE),pages1415–1428.IEEE,2025.
478
[25] XiaoLuo,JingyangYuan,ZijieHuang,HuiyuJiang,YifangQin,WeiJu,MingZhang,and
479
YizhouSun. Hope: High-ordergraphodeformodelinginteractingdynamics. InInternational
480
conferenceonmachinelearning,pages23124–23139.PMLR,2023.
481
[26] PietroPanzarasa,ToreOpsahl,andKathleenMCarley.Patternsanddynamicsofusers’behavior
482
andinteraction: Networkanalysisofanonlinecommunity. JournaloftheAmericanSocietyfor
483
InformationScienceandTechnology,60(5):911–932,2009.
484
[27] Ashwin Paranjape, Austin R Benson, and Jure Leskovec. Motifs in temporal networks. In
485
ProceedingsoftheTenthACMInternationalConferenceonWebSearchandDataMining,pages
486
601–610,2017.
487
[28] AldoPareja,GiacomoDomeniconi,JieChen,TengfeiMa,ToyotaroSuzumura,HirokiKaneza-
488
shi,TimKaler,TaoB.Schardl,andCharlesE.Leiserson. EvolveGCN:Evolvinggraphconvo-
489
lutionalnetworksfordynamicgraphs. InProceedingsoftheThirty-FourthAAAIConferenceon
490
ArtificialIntelligence,2020.
491
[29] JoonhyukPark,DonghyunLee,YujeeSong,GuorongWu,andWonHwaKim. Topology-aware
492
graph diffusion model with persistent homology. In International Conference on Learning
493
Representations(ICLR),2025. Withdrawnsubmission.
494
[30] FarimahPoursafaei,ShenyangHuang,KellinPelrine,andReihanehRabbany. Towardsbetter
495
evaluationfordynamiclinkprediction. AdvancesinNeuralInformationProcessingSystems,
496
35:32928–32941,2022.
497
[31] EmanueleRossi,BenjaminChamberlain,FabrizioFrasca,DavideEynard,FedericoMonti,and
498
MichaelBronstein. Temporalgraphnetworksfordeeplearningondynamicgraphs. ICML,
499
2020.
500
[32] AravindSankar, YanhongWu,LiangGou,WeiZhang,andHaoYang. Dysat: Deepneural
501
representationlearningondynamicgraphsviaself-attentionnetworks. InProceedingsofthe
502
13thinternationalconferenceonwebsearchanddatamining,pages519–527,2020.
503
[33] Kiarash Shamsi, Friedhelm Victor, Murat Kantarcioglu, Yulia Gel, and Cuneyt G Akcora.
504
Chartalist: Labeledgraphdatasetsforutxoandaccount-basedblockchains. AdvancesinNeural
505
InformationProcessingSystems,35:34926–34939,2022.
506
11

[34] KiarashShamsi,FarimahPoursafaei,ShenyangHuang,BaoTranGiaNgo,BarisCoskunuzer,
507
and Cuneyt Gurcan Akcora. Graphpulse: Topological representations for temporal graph
508
propertyprediction. InProceedingsofthe12thInternationalConferenceonLearningRepre-
509
sentations(Poster),2024.
510
[35] KiarashShamsi,TranGiaBaoNgo,RaziehShirzadkhani,ShenyangHuang,FarimahPoursafaei,
511
PoupakAzad,ReihanehRabbany,BarisCoskunuzer,GuillaumeRabusseau,andCuneytGurcan
512
Akcora. MiNT:Multi-networktransferbenchmarkfortemporalgraphlearning. InTheThirty-
513
ninthAnnualConferenceonNeuralInformationProcessingSystemsDatasetsandBenchmarks
514
Track,2026. URLhttps://openreview.net/forum?id=Za7IcsXIRV.
515
[36] ChristopherA.Sims. Macroeconomicsandreality. Econometrica: JournaloftheEconometric
516
Society,pages1–48,1980.
517
[37] JunweiSu,DifanZou,andChuanWu. PRES:towardscalablememory-baseddynamicgraph
518
neuralnetworks. InTheTwelfthInternationalConferenceonLearningRepresentations,ICLR
519
2024,Vienna,Austria,May7-11,2024.OpenReview.net,2024. URLhttps://openreview.
520
net/forum?id=gjXor87Xfy.
521
[38] Yuxing Tian, Yiyan Qi, and Fan Guo. Freedyg: Frequency enhanced continuous-time dy-
522
namicgraphmodelforlinkprediction. InThetwelfthinternationalconferenceonlearning
523
representations,2024.
524
[39] Astrit Tola, Funmilola Mary Taiwo, Cuneyt Gurcan Akcora, and Baris Coskunuzer. Toper:
525
Topologicalembeddingsingraphrepresentationlearning. NeurIPS,2025.
526
[40] RakshitTrivedi,MehrdadFarajtabar,PrasenjeetBiswal,andHongyuanZha. Dyrep: Learning
527
representationsoverdynamicgraphs. InInternationalconferenceonlearningrepresentations,
528
2019.
529
[41] Ronald J Williams and David Zipser. A learning algorithm for continually running fully
530
recurrentneuralnetworks. NeuralComputation,1(2):270–280,1989.
531
[42] Da Xu, Chuanwei Ruan, Evren Körpeoglu, Sushant Kumar, and Kannan Achan. Inductive
532
representation learning on temporal graphs. In 8th International Conference on Learning
533
Representations,ICLR2020,AddisAbaba,Ethiopia,April26-30,2020.OpenReview.net,2020.
534
URLhttps://openreview.net/forum?id=rJeW1yHYwH.
535
[43] Menglin Yang, Min Zhou, Marcus Kalander, Zengfeng Huang, and Irwin King. Discrete-
536
timetemporalnetworkembeddingviaimplicithierarchicallearninginhyperbolicspace. In
537
Proceedingsofthe27thACMSIGKDDConferenceonKnowledgeDiscovery&DataMining,
538
pages1975–1985,2021.
539
[44] DongshengYe,HaoJiang,YingJiang,andHaoLi. Stabledistanceofpersistenthomologyfor
540
dynamicgraphcomparison. Knowledge-BasedSystems,278:110855,2023.
541
[45] JiaxuanYou,TianyuDu,andJureLeskovec. ROLAND:graphlearningframeworkfordynamic
542
graphs. InProceedingsofthe28thACMSIGKDDConferenceonKnowledgeDiscovery&Data
543
Mining,pages2358–2366,2022.
544
[46] XingtongYu,ZhenghaoLiu,XinmingZhang,andYuanFang. Node-timeconditionalprompt
545
learningindynamicgraphs. InTheThirteenthInternationalConferenceonLearningRepresen-
546
tations,2023.
547
[47] G.UdnyYule. Onthetime-correlationproblem,withespecialreferencetothevariate-difference
548
correlationmethod. JournaloftheRoyalStatisticalSociety,84(4):497–537,1921.
549
[48] WenbinZhang,LimingZhang,DieterPfoser,andLiangZhao. Disentangleddynamicgraph
550
deepgeneration. InProceedingsofthe2021SIAMInternationalConferenceonDataMining
551
(SDM),pages738–746.SIAM,2021.
552
[49] LingZhao,YujiaoSong,ChaoZhang,YuLiu,PuWang,TaoLin,MinDeng,andHaifengLi.
553
T-gcn: Atemporalgraphconvolutionalnetworkfortrafficprediction. IEEEtransactionson
554
intelligenttransportationsystems,21(9):3848–3858,2019.
555
12

[50] YanpingZheng,LuYi,andZheweiWei. Asurveyofdynamicgraphneuralnetworks. Frontiers
556
ofComputerScience,19(6):196323,2025.
557
[51] DaweiZhou,LechengZheng,JiaweiHan,andJingruiHe.Adata-drivengraphgenerativemodel
558
fortemporalinteractionnetworks. InProceedingsofthe26thACMSIGKDDInternational
559
ConferenceonKnowledgeDiscovery&DataMining,pages401–411,2020.
560
13

Appendix
561
A BroaderImpact
562
563
TOPOGEDcanbroadentheuseoftemporalgraphlearninginsettingswheredecisionsmustbemade
beforethenextnetworkstateisobserved. Byforecastinganentireinductivesnapshot,including
564
nodearrivals,reappearances,andfutureedgestructure,itsupportsearlyriskassessment,anomaly
565
monitoring,resourceplanning,anddownstreamanalysisinevolvingnetworkssuchastransaction
566
ecosystems,onlinecommunities,andscientificinteractiongraphs. Itstopology-guideddesignmakes
567
forecastsmoreinterpretablethanisolatededgescores,sinceuserscaninspectpredictednodecounts,
568
edgebudgets, churn, andstructuralpropertiesbeforeapplyingdownstreammodels. Becausethe
569
outputisafullforecastedgraph, multipletaskscanbeevaluatedonthesamepredictedsnapshot
570
withouttask-specificretraining. Theefficientdescriptor-basedpipelinealsoreducescomputational
571
cost,makinggraphforecastingmoreaccessibleforlargedynamicnetworks.
572
B ExtendedRelatedWork
573
Temporalgraphrepresentationlearning. Temporalgraphmodelslearntime-evolvingnodeand
574
edgerepresentationstopredictlinks,nodes,orgraph-levellabels. Earlymethodsupdateembeddings
575
alonginteractionstreamswithmemory,e.g.,JODIE[23]andDyRep[40]. Morerecenttemporal
576
GNNsincorporatetemporalencodingandattentiontosupportinductivelearningovertime-stamped
577
edges,e.g.,TGAT[42]andTGN[31],whilesnapshot-basedmodelsattendoversequencesofgraphs,
578
e.g., DySAT [32]. PRES is an example of a dynamic memory model that optimizes for efficient
579
trainingbetweensnapshots[37].
580
CTDGMethods. CTDGs[38,25,45]modelthenetworkasastreamoftime-stampednode-level
581
events. Memory-basedCTDGmethodsupdateembeddingsalonginteractionstreams[23,40],while
582
temporalGNNsaddattentionandtemporalencodingforinductivelearning[42,31].
583
DTDGMethods.DTDGsrepresentthenetworkasasequenceofsnapshotgraphscollectedatdiscrete
584
timestamps. Snapshot-basedmodelsattendoversequencesofgraphs,e.g.,DySAT[32]. PRESisan
585
exampleofadynamicmemorymodelthatoptimizesforefficienttrainingbetweensnapshots[37].
586
HTGN[43]projecteachsnapshotintohyperbolicspacetomodelnoderepresentation. Whileexisting
587
methodsfocusonnode-leveltasks,GraphPulse[34]andMiNT[35]aresnapshot-basedmodelsfor
588
graphpropertypredictiontasks.
589
Dynamicgraphgeneration. Dynamicgraphgenerationhasbeenstudiedinbothdiscrete-timeand
590
continuous-timeregimes. Indiscretetime,priormethodsgeneratesequencesofsnapshotsbylearning
591
recurrentorvariationalsnapshotlatents[48,24]. Incontinuoustime,event-basedgeneratorsmodel
592
timestampedinteractions,oftenviatemporalwalksorpoint-processmachinery[9,51,10,14];while
593
effectiveforedge-eventsynthesis,theydonotdirectlyyieldcoherentsnapshotforecastsunderchurn
594
andarecommonlytiedtotrainingnodeIDs. Thisleavesagapfordiscrete-timesnapshotforecasting
595
methodsthatcaninductivelypredictfuturegraphswithnewnodeswhilemaintainingglobalstructural
596
fidelity.
597
C FurtherExperimentalDetails
598
This section provides additional experimental details omitted from the main text. Appendix C.1
599
summarizesthetemporalgraphbaselines,AppendixC.2definesthegraphconstructionmetrics,Ap-
600
601
pendixC.3reportsthedescriptorandbudgetpredictionresultsusedbyTOPOGED,andAppendixC.4
analyzescomputationalcomplexity.
602
C.1 BaselineModels
603
Inthissection,wegivefurtherdetailsaboutthetemporalgraphlearningmodelsweusedasabaseline
604
forourwork.
605
TGCN [49] is a combination of GCN and GRU. In particular, GCN is used to learn complex
606
topological structures, while GRU is used to model embedding dynamically to capture temporal
607
dependence.
608
14

HTGN[43]leveragesthepowerofhyperbolicgeometry,whichiswell-suitedforcapturinghierar-
609
chicalstructuresandcomplexrelationshipsintemporalnetworks. HTGNmapsthetemporalgraph
610
intohyperbolicspaceandutilizeshyperbolicgraphneuralnetworksandhyperbolicgatedrecurrent
611
neuralnetworkstomodeltheevolvingdynamics. Itincorporatestwokeymodulesthatarehyperbolic
612
temporalcontextualself-attention(HTA)andhyperbolictemporalconsistency(HTC)-toensurethat
613
temporaldependenciesareeffectivelycapturedandthatthemodelisbothstableandgeneralizable
614
acrossvarioustasks.
615
GCLSTM[5]combinesaGraphConvolutionalNetwork(GCN)andLongShort-TermMemory
616
unitstohandleboththestructuralandtemporalaspectsofevolvingnetworks. TheGCNisusedto
617
capturethelocalstructuralpropertiesofthenetworkateachsnapshot,whiletheLSTMlearnsthe
618
temporalevolutionofthesesnapshotsovertime.
619
EvolveGCN[28]isdesignedtocapturethetemporaldynamicsofgraph-structureddata. Insteadof
620
relyingonstaticnodeembeddings,EvolveGCNevolvestheparametersofagraphconvolutionalnet-
621
work(GCN)overtime.Byusingarecurrentneuralnetwork(RNN)toadapttheGCNparameters,this
622
modeliscapableofdynamicallyadjustingduringbothtrainingandtesting. ThismakesEvolveGCN
623
suitableforevolvinggraphswithchangingobservednodes,althoughitdoesnotexplicitlygenerate
624
unseennodesorfullfuturesnapshots.
625
ROLAND[45]isadynamicgraphlearningframeworkthatmodelsnoderepresentationsashierar-
626
chicalstates,updatedrecurrentlytocapturetemporaldependenciesinevolvinggraphs. Itsupports
627
scalabletrainingusingtechniquesliketruncatedbackpropagationthroughtimeandmeta-learning. In
628
ourDTDGsetting,weuseROLANDtobenchmarkitsperformanceandadaptabilityacrossdiverse
629
temporalnetworks.
630
VGRNN[11]introducesahierarchicalvariationalmodelfordynamicgraphs,extendingthevari-
631
ational autoencoder to capture temporal dependencies. The framework utilizes high-level latent
632
random variables within a graph recurrent neural network to model evolving topology and node
633
attributesovertime.
634
C.2 EvaluationMetrics
635
636
WeevaluateeachpredictedsnapshotG(cid:98)t+1 =(V(cid:98)t+1 ,E(cid:98)t+1 )againstthecorrespondingground-truth
snapshotG =(V ,E )overtheN testtimestamps.Ourmetricsmeasurethreecomplementary
637 t+1 t+1 t+1
aspectsofforecastquality: structuralfidelity,nodeandedgerecovery,andbudgeterrors.
638
Structuralfidelity. Foragraphstatisticf,wereportthemeanabsoluterelativeerror
639
(cid:12) (cid:12)
S = 1 (cid:88) N (cid:12) (cid:12) f(G(cid:98)t+1 )−f(G t+1 )(cid:12) (cid:12),
f N (cid:12) f(G )+ε (cid:12)
(cid:12) t+1 (cid:12)
t=1
where ε > 0 is a small constant used to avoid division by zero. Lower values indicate closer
640
agreementbetweenthepredictedandtruesnapshots. Weapplythisevaluationtoaveragenodedegree,
641
numberofuniquedegrees,averagedegreecentrality,assortativitycoefficient,clusteringcoefficient,
642
density,numberoftriangles,andthefiltration-descriptorerror. Forthedescriptormetric,bothgraphs
643
areencodedusingthedegree-basedfiltrationdescriptorΦ(·),andwereporttherelativeerrorbetween
644
theresultingdescriptorvectors.
645
Nodeandedgerecovery. Weevaluaterecoveryofthetruenodeandedgesetsusingprecision,recall,
646
647
andF1scores. Fornodes,thesemetricscomparethepredictedactivenodesetV(cid:98)t+1 withthetrue
648
activenodesetV
t+1
. Foredges,theycompareE(cid:98)t+1 withE
t+1
,eitheroveralledgesorwithinthe
inductiveedgecategoriesold–old-bank,old–old-nobank,old–new,andnew–new. Aggregatescores
649
arecomputedbyaveragingthecorrespondingmetricoveralltestsnapshots:
650
N
1 (cid:88)
S
f
=
N
f(G(cid:98)t+1 ,G
t+1
).
t=1
Forthesemetrics,highervaluesindicatebetterrecovery.
651
Budgeterrors. Wealsoreportmedianerrorsinthepredictednodeandedgebudgets. Thesemetrics
652
quantifyover-generationandunder-generationseparately:
653
ExtraNodes
t
=max{|V(cid:98)t+1 |−|V
t+1
|,0}, MissingNodes
t
=max{|V
t+1
|−|V(cid:98)t+1 |,0},
15

654 and
ExtraEdges =max{|E(cid:98)t+1 |−|E |,0}, MissingEdges =max{|E |−|E(cid:98)t+1 |,0}.
|     | t   |     | t+1 |     | t t+1 |
| --- | --- | --- | --- | --- | ----- |
Wesummarizeeachquantitybyitsmedianovertestsnapshots:
655
(cid:0) (cid:1)
|     |     |     | S =median | {f }N | .   |
| --- | --- | --- | --------- | ----- | --- |
|     |     |     | f         | t t=1 |     |
656 Lowervaluesindicatebettercalibrationofthepredictedgraphsize. Together,thesemetricsassess
657 whetheramodelreconstructsthecorrectglobalstructure,recoversthecorrectactivenodesandedges,
andavoidsexcessiveover-orunder-generation.
658
| 659 C.3 DescriptorPredictionPerformance |     |     |     |     |     |
| --------------------------------------- | --- | --- | --- | --- | --- |
660 Topredictthedescriptorandprobabilityvectorsusedtoguidegraphconstruction,wetestedmultiple
prediction methods. We tested the models SMA [47], VAR [36], V-EWMA [2], SSM [17], and
661
VECM[16].
662
663 TheperformanceisdetailedinTables6and7. Asthetablesshow,V-EWMAhasthebestaverage
664 performance,achievingthelowestmeanerrorforbothnodeandedgeprediction,withaverageerrors
665 of8.12%and10.82%,respectively. ThisindicatesthatV-EWMAprovidesthemostaccuratejoint
predictionofthedescriptorandprobabilityvectorsusedingraphconstruction.
666
|     |              |     | Table6: PercentErrorNodes |        |              |
| --- | ------------ | --- | ------------------------- | ------ | ------------ |
|     | Dataset      |     | SMA VAR                   | V-EWMA | SSM VECM     |
|     | Collegemsg   |     | 15.10 23.43               | 11.64  | 27.88 18.94  |
|     | Mathoverflow |     | 0.28 -1.27                | 0.41   | -3.65 -1.76  |
|     | Adex         |     | 7.41 60.63                | 9.84   | 11.23 44.21  |
|     | Aeternity    |     | -1.64 17.88               | 0.85   | 3.42 -0.81   |
|     | Aion         |     | 0.02 10.08                | -0.27  | 22.31 13.67  |
|     | Aragon       |     | 6.69 47.72                | 5.51   | 18.55 39.06  |
|     | Bancor       |     | 0.28 46.64                | 1.31   | 5.42 27.22   |
|     | Centra       |     | 43.65 63.00               | 37.33  | 64.64 57.76  |
|     | Cindicator   |     | 5.85 60.88                | 4.52   | 40.49 48.90  |
|     | Coindash     |     | 1.89 119.84               | 8.42   | 15.36 117.00 |
4.95
|     | Dgd |     | 14.15 | 10.85 | 22.94 9.20 |
| --- | --- | --- | ----- | ----- | ---------- |
15.62
|                                     | Iconomi   |     | 17.30 26.76 |      | 34.18 24.04 |
| ----------------------------------- | --------- | --- | ----------- | ---- | ----------- |
|                                     | Reddit_b  |     | 0.84 -3.24  | 0.67 | 0.65 -3.86  |
|                                     | Tgbl-wiki |     | 9.13 5.32   | 6.96 | 7.26 6.44   |
|                                     | Average   |     | 8.64 34.47  | 8.12 | 19.33 28.57 |
| C.4 ComputationalComplexityAnalysis |           |     |             |      |             |
667
Letn =|V |andm =|E |denotethenumberofnodesandedgesintheobservedsnapshot,and
| 668 t | t t | t   |     |     |     |
| ----- | --- | --- | --- | --- | --- |
let n = |V(cid:98)t+1 | and m = |E(cid:98)t+1 | denotethe predictednodeand edgebudgets. Let d bethe
| 669 (cid:98)t+1 |     | (cid:98)t+1 |     |     |     |
| --------------- | --- | ----------- | --- | --- | --- |
embedding dimension, L the number of GCN layers, and M the number of sampled candidate
670
partnerspernode.
671
672 Thefiltrationencodercomputesdegreesandcumulativenodeandedgecountsoverdegreethresholds,
whichcostsO(n +m )persnapshotoncethethresholdsetisfixed.Thedescriptorpredictoroperates
| 673 | t t |     |     |     |     |
| --- | --- | --- | --- | --- | --- |
onafixed-lengthsequenceoffiltrationvectors,soitscostdoesnotscalewiththegraphsize. The
674
memorymodulescansthelastksnapshotstorecovercandidateoldnodesandcomputereappearance
675
(cid:80)t
676 scores,withworst-casecostO( m ). Thisavoidsscoringthefullhistoricalnodeuniverse.
|     |     |     | τ=t−k τ |     |     |
| --- | --- | --- | ------- | --- | --- |
Thedecoderisthedominantterm. Eachdecodingphasescoresarestrictedcandidatesetinsteadof
677
allpossiblenodepairs. WithatmostM candidatepartnersperpredictednode,candidatescoring
678
costsO(Mn s(d)),wheres(d)isthecostofscoringonecandidateedge. Selectingthetopedges
| 679 | (cid:98)t+1 |     |     |     |     |
| --- | ----------- | --- | --- | --- | --- |
16

|     |              | Table7:     | PercentErrorEdges |       |        |
| --- | ------------ | ----------- | ----------------- | ----- | ------ |
|     | Dataset      | SMA VAR     | V-EWMA            | SSM   | VECM   |
|     | Collegemsg   | 29.06 41.57 | 23.65             | 46.19 | 20.99  |
|     | Mathoverflow | 1.11 -1.25  | 1.23              | -3.22 | -1.38  |
|     | Adex         | 12.25 71.53 | 16.57             | 14.23 | 55.26  |
|     | Aeternity    | -2.21 13.14 | -0.01             | 2.02  | -2.08  |
|     | Aion         | 0.75        | 7.79 0.49         | 19.71 | 10.93  |
|     | Aragon       | 7.48 52.68  | 6.25              | 19.80 | 43.73  |
|     | Bancor       | 1.50 45.89  | 2.93              | 6.45  | 27.41  |
|     | Centra       | 51.01 56.46 | 41.90             | 80.14 | 65.42  |
|     | Cindicator   | 6.42 48.65  | 5.11              | 40.96 | 37.83  |
|     | Coindash     | 3.67 103.73 | 10.09             | 18.76 | 112.69 |
|     | Dgd          | 18.87       | 9.40 14.79        | 29.81 | 13.72  |
|     | Iconomi      | 20.77 26.55 | 18.55             | 37.92 | 23.83  |
|     | Reddit_b     | 2.64 -4.84  | 2.10              | 2.70  | -5.02  |
|     | Tgbl-wiki    | 10.55       | 6.88 7.88         | 8.42  | 9.59   |
|     | Average      | 11.70 34.16 | 10.82             | 23.13 | 29.49  |
addsO(Mn logm ). RecomputingembeddingswithanL-layersparseGCNoverthepartially
| 680 | (cid:98)t+1 (cid:98)t+1 |     |     |     |     |
| --- | ----------------------- | --- | --- | --- | --- |
generatedgraphcostsO(L(m d+n d2))acrossaconstantnumberofdecodingphases.
| 681 | (cid:98)t+1 | (cid:98)t+1 |     |     |     |
| --- | ----------- | ----------- | --- | --- | --- |
(cid:80)t
682 Thus,theper-snapshotinferencecostisO(n t +m t + m τ +Mn (cid:98)t+1 (s(d)+logm (cid:98)t+1 )+
τ=t−k
L(m d+n d2)). Forfixedk,M,L,anddescriptorlength,thecostscalesnear-linearlyinthe
| 683 (cid:98)t+1 | (cid:98)t+1 |     |     |     |     |
| --------------- | ----------- | --- | --- | --- | --- |
observedmemorywindowandthepredictedsnapshotsize. Traininghasthesameleadingterms,with
684
685 theedge-scoringcostmultipliedby(1+r)whenrnegativeedgesaresampledperpositiveedge.
| D Forecast-to-GraphReconstructionDetails |     |     |     |     |     |
| ---------------------------------------- | --- | --- | --- | --- | --- |
686
ThissectionprovidestheimplementationdetailsforconvertingthequantitiesforecastbyTOPOGED
687
688 intoaconcretefuturesnapshotG(cid:98)t+1 = (V(cid:98)t+1 ,E(cid:98)t+1 ). Wefirstdescribethedegree-basedfiltration
689 descriptoranditsinversereconstructionroleinAppendixD.1. Wethendetailhowthepredicted
node set is assembled, including reappearing-node sampling from a recent memory window in
690
AppendixD.2anddegree-binassignmentforcold-startnodesinAppendixD.3. Next, wedefine
691
theedge-typeprobabilitiesusedtoallocatethepredictededgebudgetacrossold–old-bank,old–old-
692
693 nobank, old–new, and new–new interactions in Appendix D.4. These components are combined
694 inthefullbudgetedmulti-phasereconstructionalgorithminAlgorithm1. Finally,AppendixD.5
explainshowlink-predictionbaselinesareadaptedtothesamefull-graphconstructionsettingthrough
695
thresholdingandboundedadjacencyreconstruction.
696
| D.1 Degree-BasedFiltrationEncoderandInverseTopologyDecoder |     |     |     |     |     |
| ---------------------------------------------------------- | --- | --- | --- | --- | --- |
697
LetG =(V ,E )denoteagraphsnapshotattimet,whereV isthenodesetandE istheedgeset.
| 698 t | t t |     |     | t   | t   |
| ----- | --- | --- | --- | --- | --- |
Weencodeeachsnapshotbyafiltration-basedmultiscaletopologicaldescriptor. Thegoalofthis
699
descriptoristosummarizetheevolutionofthegraphstructureacrossincreasingdegreethresholds,
700
701 whileretainingenoughinformationtolaterreconstructagraphsnapshotconsistentwiththeencoded
702 topology.
Weusethedegreefunction
703
|                          |                       | f (v)=deg | (v), | v ∈V , |     |
| ------------------------ | --------------------- | --------- | ---- | ------ | --- |
|                          |                       | t         | Gt   | t      |     |
| asthefiltrationfunction. | Givendegreethresholds |           |      |        |     |
704
|                                                  |     | ϵ <ϵ | <···<ϵ | ,   |     |
| ------------------------------------------------ | --- | ---- | ------ | --- | --- |
|                                                  |     | 1    | 2      | n   |     |
| 705 wedefinetheactivevertexsetatfiltrationlevelϵ |     |      | by     |     |     |
i
|     |     | V ={v | ∈V :f (v)≤ϵ | }.  |     |
| --- | --- | ----- | ----------- | --- | --- |
|     |     | t,ϵi  | t t         | i   |     |
17

706 Thecorrespondingfilteredgraphistheinducedsubgraph
|     | G =G | [V ]=(V | ,E ),     |     |
| --- | ---- | ------- | --------- | --- |
|     | t,ϵi | t t,ϵi  | t,ϵi t,ϵi |     |
where
707
|     | E ={(u,v)∈E | :u,v | ∈V }. |     |
| --- | ----------- | ---- | ----- | --- |
|     | t,ϵi        | t    | t,ϵi  |     |
Thus,thefiltrationproducesanestedsequenceofinducedsubgraphs
708
|     | G ⊆G | ⊆···⊆G | =G .   |     |
| --- | ---- | ------ | ------ | --- |
|     | t,ϵ1 | t,ϵ2   | t,ϵn t |     |
709 Ateachfiltrationlevel,werecordtwostructuralquantities: thenumberofactiveverticesandthe
| 710 numberofinducededges. | Specifically,wedefine |          |        |     |
| ------------------------- | --------------------- | -------- | ------ | --- |
|                           | x =|V                 | |, y     | =|E |. |     |
|                           | t,i                   | t,ϵi t,i | t,ϵi   |     |
Thisgivestwomonotonesequences
711
|     | X =(x | ,x ,...,x | )   |     |
| --- | ----- | --------- | --- | --- |
|     | t     | t,1 t,2   | t,n |     |
and
712
|                                                        | Y =(y | ,y ,...,y | ).              |     |
| ------------------------------------------------------ | ----- | --------- | --------------- | --- |
|                                                        | t     | t,1 t,2   | t,n             |     |
| Thefiltration-basedtopologicaldescriptorofthesnapshotG |       |           | isthendefinedas |     |
| 713                                                    |       |           | t               |     |
|                                                        | Φ(G   | ):=(X ,Y  | ).              |     |
|                                                        |       | t t       | t               |     |
714 Byconstruction,thefinalfiltrationlevelrecoversthesizeoftheoriginalsnapshot:
|     | x =|V | |, y  | =|E |. |     |
| --- | ----- | ----- | ------ | --- |
|     | t,n   | t t,n | t      |     |
715 Therefore,Φ(G )providesacompactmultiscalerepresentationofthesnapshot,recordinghowthe
t
numberofverticesandinducededgesgrowsasthedegreethresholdincreases.
716
Theuseofthedegreefiltrationhastwoadvantages. First,degreeiscomputationallyinexpensiveand
717
directlyavailablefromthegraphstructure. Second,andmoreimportantly,degreeprovidesanatural
718
719 handleontheinversetopologyproblem. Intheforwarddirection,theencodermapsagraphsnapshot
720 toitsdescriptor,
|     | G (cid:55)−→Φ(G | )=(X | ,Y ). |     |
| --- | --------------- | ---- | ----- | --- |
|     | t               | t    | t t   |     |
Theinversetopologyproblemasksforthereverseoperation: givenapredicteddescriptor
721
|     | Φ(cid:98)t+1 | =(X(cid:98)t+1 ,Y(cid:98)t+1 | ),  |     |
| --- | ------------ | ---------------------------- | --- | --- |
722 constructagraphsnapshot
|                                                | G(cid:98)t+1 | =(V(cid:98)t+1 ,E(cid:98)t+1 | )   |     |
| ---------------------------------------------- | ------------ | ---------------------------- | --- | --- |
| whosefiltrationrealizesthepredicteddescriptor. |              | Thatis,weseek                |     |     |
723
|                                |                                   | Φ(G(cid:98)t+1 )≈Φ(cid:98)t+1 | .             |     |
| ------------------------------ | --------------------------------- | ----------------------------- | ------------- | --- |
| Equivalently,foreachthresholdϵ | ,theinducedsubgraphofG(cid:98)t+1 |                               | shouldsatisfy |     |
| 724                            | i                                 |                               |               |     |
|                                | |V(cid:98)t+1,ϵi |=x              | , |E(cid:98)t+1,ϵi            | |=y .         |     |
|                                |                                   | (cid:98)t+1,i                 | (cid:98)t+1,i |     |
Thisinverseproblemshouldbeunderstoodasagraphrealizationproblemratherthanexactgraph
725
recovery. Thedescriptor(X ,Y )doesnotencodethefulladjacencymatrixofG ;instead,itrecords
| 726 | t t |     |     | t   |
| --- | --- | --- | --- | --- |
727 the number of active vertices and induced edges at each degree threshold. Consequently, many
728 non-isomorphicgraphsmaysharethesamedescriptor. Thedecoderthereforeaimstoconstructone
graphthatisconsistentwiththepredictedmultiscaletopology.
729
Anaturaldecodingprocedureproceedslevelbylevelthroughthepredictedfiltration. Supposethe
730
predicteddescriptoris
731
|     | X(cid:98) =(x ,...,x | ), Y(cid:98) | =(y ,...,y ).       |     |
| --- | -------------------- | ------------ | ------------------- | --- |
|     | (cid:98)1            | (cid:98)n    | (cid:98)1 (cid:98)n |     |
Atthefirstfiltrationlevel,thedecodercreatesx verticesandaddsy inducededgesamongthem. At
| 732 |     | (cid:98)1 | (cid:98)1 |     |
| --- | --- | --------- | --------- | --- |
thenextlevel,itadds
733
x −x
(cid:98)2 (cid:98)1
18

| | |=2,| | |=0 |     | | |=3,| | |=1 | | |=5,| | |=1 | | |=8,| | |=11 |
| ------- | --- | --- | ------- | --- | ------- | --- | ------- | ---- |
|         |     |     |         |     |         | 5   | 6       | 5    |
| 2       |     | 1 2 |         |     | 1 2     | 1   | 2 8     | 1    |
|         |     |     |         |     | 3 4     | 3   | 4       | 7 3  |
Figure6: Degree-thresholdfiltrationofagraphsnapshot. Ateachfiltrationlevelϵ , theinduced
i
subgraphrecordsthenumberofactivevertices|V|andinducededges|E|.
734 newverticesandthenaddsedgessothattheinducededgecountbecomesy (cid:98)2 . Continuinginthisway,
735 atlevelithedecoderadds
x −x
(cid:98)i (cid:98)i−1
736 newverticesandintroduces
y (cid:98)i −y (cid:98)i−1
additionalinducededges,subjecttotheconstraintthatallpreviouslyconstructedsubgraphsremain
737
unchanged. Thus,thepredicteddescriptorcontrolsbothnodeinclusionandedgegrowthacrossthe
738
739 filtration.
740 Formally,thedecoderconstructsnestedvertexsets
|     |     |     | V(cid:98)ϵ1 | ⊆V(cid:98)ϵ2 | ⊆···⊆V(cid:98)ϵn =V(cid:98) |     |     |     |
| --- | --- | --- | ----------- | ------------ | --------------------------- | --- | --- | --- |
with
741
|     |     |     |     | |V(cid:98)ϵi | |=x , |     |     |     |
| --- | --- | --- | --- | ------------ | ----- | --- | --- | --- |
(cid:98)i
742 andedgesetssatisfying
|     |     | |E(cid:98)ϵi | |=y , | E(cid:98)ϵi ={(u,v)∈E(cid:98):u,v |     | ∈V(cid:98)ϵi }. |     |     |
| --- | --- | ------------ | ----- | --------------------------------- | --- | --------------- | --- | --- |
(cid:98)i
743 Thefinaloutputisthegraph
G(cid:98)=(V(cid:98),E(cid:98)),
744 where
|     |     |     |     | |V(cid:98)|=x | , |E(cid:98)|=y | .   |     |     |
| --- | --- | --- | --- | ------------- | --------------- | --- | --- | --- |
|     |     |     |     | (cid:98)n     | (cid:98)n       |     |     |     |
Thisconstructionmakesthedescriptordirectlyactionable. ThesequenceX(cid:98) determineshowmany
745
746 verticesmustbeintroducedateachdegreethreshold,whilethesequenceY(cid:98) determineshowmany
747 inducededgesmustbepresentamongthosevertices. Therefore,thedegree-baseddescriptorisnot
onlyarepresentationofthegraph;italsoservesasablueprintforconstructingagraphsnapshotfrom
748
predictedtopologicaldata.
749
Figure6illustratesthisprocess. ForagraphG = (V,E),thefiltrationlevelsϵ ,ϵ ,ϵ ,ϵ produce
| 750 |     |     |     |     |     |     | 1 2 3 | 4   |
| --- | --- | --- | --- | --- | --- | --- | ----- | --- |
751 inducedsubgraphswithdescriptors
|     | (|V | |,|E | |), (|V | |,|E |), | (|V |,|E | |), (|V |,|E | |). |     |
| --- | --- | ---- | ------- | -------- | -------- | ------------ | --- | --- |
|     |     | ϵ1   | ϵ1      | ϵ2 ϵ2    | ϵ3 ϵ3    | ϵ4 ϵ4        |     |     |
Inthedisplayedexample,thesevaluesare
752
|     |     |     | (2,0), | (3,1), | (5,1), | (8,11). |     |     |
| --- | --- | --- | ------ | ------ | ------ | ------- | --- | --- |
753 Thus,
|     |     |     | X =(2,3,5,8), |     | Y =(0,1,1,11). |     |     |     |
| --- | --- | --- | ------------- | --- | -------------- | --- | --- | --- |
ReadingthefiltrationforwardencodesthegraphintothedescriptorΦ(G)=(X,Y),whilereading
754
755 itinreverseshowshowthesamedescriptorcanbeusedtodecodeagraphrealization. Thelarge
increasefrom|E |=1to|E |=11indicatesthatmostoftheedgestructureappearswhenthefinal
| 756 | ϵ3  |     | ϵ4  |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
groupofverticesentersthefiltration. Hencethefiltrationdescriptorcapturesnotonlythefinalsizeof
757
thegraph,butalsothescaleatwhichitsconnectivityemerges.
758
Insummary,theencodermapseachsnapshotG
| 759 |     |     |     |     | t toacompactdegree-basedtopologicaldescriptor |     |     |     |
| --- | --- | --- | --- | --- | --------------------------------------------- | --- | --- | --- |
760 Φ(G t )=(X t ,Y t ). Theinversetopologydecoderthenusesapredicteddescriptortoconstructagraph
snapshot whose thresholded induced subgraphs match the predicted node and edge counts. This
761
providesadirectbridgebetweentopologicalpredictionandgraphgeneration: insteadofpredicting
762
every edge independently, the model predicts a multiscale structural summary, and the decoder
763
realizesthissummaryasaconcretegraph.
764
19

| 765 D.2 | NodeSetPredictionandReappearanceSampling |     |     |     |     |     |     |     |
| ------- | ---------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
766 A primary contribution of TOPOGED is the node prediction heuristic for old nodes. For each
timestampT,insteadofpredictingedgeswiththeentirenodeuniverseforthedataset,weassign
767
nodesa’reappearanceprobability’foraweightedrandomsampling.
768
| Foreachv | ∈V  | ,weassign |            |     |              |                                |     |     |
| -------- | --- | --------- | ---------- | --- | ------------ | ------------------------------ | --- | --- |
| 769      |     | t−k:t     |            |     |              |                                |     |     |
|          |     |           |            |     | (cid:18) t−ℓ | (cid:19)(cid:18) deg (cid:19)α |     |     |
|          |     |           |            |     |              | v v                            | fβ, |     |
|          |     |           | P(v,t)=exp |     | −            |                                |     | (2) |
|          |     |           |            |     | λ            | deg                            | v   |     |
max
whereℓ isthelastobservedtimeofv,deg isitsdegreeaggregatedoverthewindow,deg is
| 770 | v   |     |     |     | v   |     |     | max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
themaximumdegreeinthewindow,andf isthenumberofsnapshotsinwhichvappearswithin
| 771 |     |     |     |     | v   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
the window. The exponential term enforces recency, the normalized degree captures structural
772
prominence,andthefrequencytermcapturespersistenceacrosssnapshots. Wekeepthesefactors
773
774 separatetoavoidconflatinginstantaneousconnectivitywithtemporalactivity. Thenormalizationby
| 775 deg | keepsscoresonacomparablescaleacrossdatasets. |     |     |     |     |     |     |     |
| ------- | -------------------------------------------- | --- | --- | --- | --- | --- | --- | --- |
max
Accordingtothedescriptorandprobabilityvector,nnodesarerandomlysampledandaddedtothe
776
currentgraphtobeconstructed.
777
778 Weightedsamplingpreventsthedecoderfromalwaysselectingthesamehighest-scorenodes,while
779 stillfavoringnodeswithstrongerrecency,degree,andfrequencyevidence.
Tables8and9summarizethefittedparametersandtheresultingnodereappearanceperformance.
780
Theoptimalβ valuesareconsistentlyhighacrossdatasets, showingthatrepeatedactivitywithin
781
the memory window is the strongest signal for old-node reappearance. The α values vary more
782
783 substantially,whichindicatesthatdegreeisusefulbutdataset-dependent. Thefittedλvaluesare
784 generallybelow1,sorecentactivityreceivesmuchlargerweightthanolderactivityinmostdatasets.
TheF1scoresshowthatnodereappearanceishighlydataset-dependent.TGBL-WikiandCollegeMsg
785
achievethestrongesttestperformance,whilesparsetokengraphssuchasAionandAragonremain
786
difficult. Inseveraldatasets,validationandtestperformanceexceedtrainingperformance,suggesting
787
788 thattheheuristicisnotsimplymemorizingthetrainingsnapshots. Overall, theseresultssupport
789 theuseofasmallmemorymodule: recency,degree,andfrequencyprovideenoughsignaltoselect
plausibleoldnodeswithoutscoringthefullhistoricalnodeuniverse.
790
Table8: OptimalParameters Table9: NodereappearancepredictionF1Score
|     | Dataset    |     | α β       | λ    |     | Dataset    | Train Val     | Test   |
| --- | ---------- | --- | --------- | ---- | --- | ---------- | ------------- | ------ |
|     | CollegeMsg |     | 3.88 8.81 | 0.31 |     | CollegeMsg | 0.4563 0.4366 | 0.3976 |
|     | Adex       |     | 2.64 8.94 | 0.85 |     | Adex       | 0.0915 0.1637 | 0.2257 |
|     | Aeternity  |     | 3.53 9.24 | 0.53 |     | Aeternity  | 0.1276 0.2103 | 0.2353 |
|     | Aion       |     | 1.94 8.75 | 0.60 |     | Aion       | 0.0645 0.0917 | 0.0944 |
|     | Aragon     |     | 3.53 9.99 | 0.55 |     | Aragon     | 0.0719 0.0670 | 0.1128 |
|     | Bancor     |     | 4.19 9.51 | 0.64 |     | Bancor     | 0.1402 0.1710 | 0.2383 |
|     | Centra     |     | 3.25 9.78 | 0.35 |     | Centra     | 0.1230 0.2380 | 0.1723 |
|     | Cindicator |     | 3.74 9.24 | 0.51 |     | Cindicator | 0.0655 0.1197 | 0.1491 |
|     | Coindash   |     | 2.97 9.47 | 0.63 |     | Coindash   | 0.1311 0.2483 | 0.2866 |
|     | Dgd        |     | 1.32 9.96 | 0.31 |     | Dgd        | 0.2203 0.1999 | 0.2428 |
|     | Iconomi    |     | 2.94 9.70 | 0.37 |     | Iconomi    | 0.1306 0.1136 | 0.2212 |
mathoverflow 5.94 7.46 0.15 mathoverflow 0.3160 0.3125 0.2993
|     | Reddit_B                               |     | 3.92 9.96 | 0.96 |     | Reddit_B  | 0.2511 0.2556 | 0.2777 |
| --- | -------------------------------------- | --- | --------- | ---- | --- | --------- | ------------- | ------ |
|     | TGBL-Wiki                              |     | 4.67 9.32 | 0.21 |     | TGBL-Wiki | 0.5542 0.5608 | 0.5525 |
| D.3 | NodeBinningandCold-StartInitialization |     |           |      |     |           |               |        |
791
For nodes v ∈ V for nodeset V for graph G , nodes are assigned a maximum degree in
| 792 |     | t+1 |     |     |     | t+1 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
accordancewiththecurrentfiltrationsequence.
793
Previouslyexistingnodes,v ∈V areassignedamaximumdegreeequivalenttotheirmostrecent
old
| appearance,roundeduptothresholdϵ |     |     |              | k   | forthenearestnon-emptybin. |              |     |     |
| -------------------------------- | --- | --- | ------------ | --- | -------------------------- | ------------ | --- | --- |
|                                  |     |     | deg(v)=min{ϵ |     | ∈B |ϵ                      | ≥deg(v) andV | >0} |     |
|                                  |     |     |              |     | k k                        | τ            | k   |     |
20

| Table10: | Glossaryofcomplexityvariables. |     |     |     |
| -------- | ------------------------------ | --- | --- | --- |
Symbol Meaning
T Totaltrainingepochs,fixedat100
N Numberoftrainingsnapshots,usingthe70%trainingpartition
train
N toN Testsetrange,usingthefinal15%partition
val
L NumberofGCN/GATlayersintheencoder
d Nodeembeddingdimensionality
H Predictiveheadsforedgetypes{o-o,o-n,n-n}
|E+ | Numberofpositiveedgesamplesperheadduringtraining
i,h
|E−
| Numberofnegativeedgesamplesperheadduringtraining
i,h
E t ,V t Observededgesandnodesintheinputcontextsnapshot
E(cid:98),V(cid:98) TargetedgesandnodesdefinedbyTopERandprobabilityvectors
B Numberoffiltrationbucketsusedfornode-degreeassignment
M Numberofpotentialedgecandidatessampledpernode
s(d) ComputationalcostoftheMLPscoringfunction
logE(cid:98) Complexitytermfortop-Kedgeselectionthroughsorting
794 Whereϵ k isthethresholdforeachbininB,deg(v) τ isthemostrecentlyseendegreefornodev,and
795 V isthenumberofnodesinbink.
k
Forv ∈V ,newnodesinitiallyhavenoinformation. Therefore,foreachnode,nodesarerandomly
new
assignedamaximumdegreeϵifthebinisnotempty.
| deg(v)=random({ϵ |     | ∈B  | |V >0}) |     |
| ---------------- | --- | --- | ------- | --- |
|                  |     | k   | k       |     |
D.4 Edge-TypeCountsandProbabilities
796
797 WestudytherelativeprevalenceofinductiveedgetypesinsnapshotG t =(V t ,E t )usingthedecom-
798 positiondefinedinSection3. Recallthatnodesarepartitionedintooldandnewateachtimesteptas
V = Vold∪Vnew,andedgesaregroupedintofourdisjointtypes: old–old-bank(recurringedges
799 t t t
betweenoldnodes),old–old-nobank(newedgesbetweenoldnodes),old–new(edgesconnectingold
800
andnewnodes),andnew–new(edgesbetweennewnodes).
801
802 Thetotalnumberofedgesattimetis:
|E |=|Eoo-bank|+|Eoo-nobank|+|Eon|+|Enn|.
| t   | t   | t   | t   | t   |
| --- | --- | --- | --- | --- |
Wedefinethenormalizededge-typeprobabilitiesastheshareofeachtypeinthesnapshot:
803
|          | |Eoo-bank| |              | |Eoo-nobank| |     |
| -------- | ---------- | ------------ | ------------ | --- |
| πoo-bank | = t        | , πoo-nobank | = t          | ,   |
| t        | |E |       | t            | |E           | |   |
|          | t          |              | t            |     |
|          | |Eon|      | |Enn|        |              |     |
| πon      | = t ,      | πnn =        | t .          |     |
| t        |            | t            |              |     |
|          | |E |       | |E           | |            |     |
|          | t          |              | t            |     |
Theseprobabilitiesquantifytheinductiveedge-typecompositionofthecurrentsnapshot.Forexample,
804
pbank(t)measurestherecurrencerateofpreviouslyobservededges,whilep (t)reflectspurelynovel
805 oo nn
interactionsamongnewlyarrivingnodes.
806
21

Algorithm1:TOPOGEDInference: BudgetedMulti-PhaseSnapshotReconstruction
Input: Observedsnapshots{G ,...,G };EdgeSetB;encoderΦ(·);topologypredictor
|     |                                  |     |     | t−k+1 |     | t   |     |     |     |
| --- | -------------------------------- | --- | --- | ----- | --- | --- | --- | --- | --- |
|     | Pred;trunkencoderEnc;typeheads{M |     |     |       |     | }   | .   |     |     |
z z∈Z
| Output: | PredictedsnapshotG(cid:98)t+1 |     |     | =(V(cid:98)t+1 | ,E(cid:98)t+1 | )andupdatedEdgeBankB. |     |     |     |
| ------- | ----------------------------- | --- | --- | -------------- | ------------- | --------------------- | --- | --- | --- |
1 1. GlobalBudgetForecasting.
| 2 ComputeΦ(G |     | )forτ | =t−k+1,...,t. |     |     |     |     |     |     |
| ------------ | --- | ----- | ------------- | --- | --- | --- | --- | --- | --- |
τ
n ew
| 3 PredictΦ(cid:98)(G | t+1             | ), |V(cid:98)t+1 | |, E(cid:98)t+1 | , p (cid:98)   | andedge-typeproportionsπ |     |                  | t+1 usingPred. |     |
| -------------------- | --------------- | ---------------- | --------------- | -------------- | ------------------------ | --- | ---------------- | -------------- | --- |
|                      |                 |                  |                 | t+ 1           |                          |     |                  |                |     |
| SetV(cid:98) o ld    | =⌊|V(cid:98)t+1 | |(1−p            | n ew            | )⌋andV(cid:98) | n ew =|V(cid:98)t+1      |     | |−V(cid:98) o ld |                |     |
| 4 t +                | 1               |                  | (cid:98) t+ 1   |                | t + 1                    |     | t + 1 .          |                |     |
Setper-typeedgebudgets c (t+1)=⌊π (t+1)E(cid:98)t+1 ⌋forz ∈{oo-bank,oo-nobank,on,nn}.
| 5   |     |     | (cid:98) z |     | z   |     |     |     |     |
| --- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- |
6 2. InductiveNodeSetAssembly.
(cid:83)t
7 ReappearingNodes: SampleN(cid:98)old nodesfromthehistoricaluniverse V using
|     |     |     |     |     |     |     |     | τ=t−k+1 | τ   |
| --- | --- | --- | --- | --- | --- | --- | --- | ------- | --- |
∈V(cid:98)old
weightedrandomsamplingasinSection4.3. Initializev withtheirmostrecent
t+1
observeddegree.
Cold-StartNodes: InitializeN(cid:98)new nodeidentifiers. Assigneachv ∈V(cid:98) n ew aninitialdegreevalue
| 8                                                           |     |     |     |     |     |     |     | t + 1 |     |
| ----------------------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ----- | --- |
| fromtheremainingcapacityinthepredicteddescriptorΦ(cid:98)(G |     |     |     |     |     |     |     | ).    |     |
t+1
| 9 DefineV(cid:98)t+1 | =V(cid:98) | o ld ∪V(cid:98) | n ew . |     |     |     |     |     |     |
| -------------------- | ---------- | --------------- | ------ | --- | --- | --- | --- | --- | --- |
|                      |            | t + 1           | t + 1  |     |     |     |     |     |     |
10 3. RepresentationLearning.
| 807 |     |     |     | o ld |     |     |     |     |     |
| --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- |
11 Initializeembeddingsh: forv ∈V(cid:98) ,usethemostrecentlyseenembeddingfromhistory;for
|     |     |     |     | t + | 1   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
v ∈V(cid:98)new,initializewithzerovectors.
t+1
| Computeinductiverepresentationsh←Enc(V(cid:98)t+1 |     |     |     |     |     |     | ,G ).   |     |     |
| ------------------------------------------------- | --- | --- | --- | --- | --- | --- | ------- | --- | --- |
| 12                                                |     |     |     |     |     |     | t−k+1:t |     |     |
13 4. Multi-PhaseEdgeReconstruction(Top-KFiltering).
o ld
14 PhaseI(BankRecurrence): DefineC oo-bank ={(u,v)∈B |u,v ∈V(cid:98) }. Addtop (cid:98) c oo-bank
t + 1
| pairsscoredbyM |     |         | (h  | ,h )toE(cid:98)t+1 | .   |     |     |     |     |
| -------------- | --- | ------- | --- | ------------------ | --- | --- | --- | --- | --- |
|                |     | oo-bank | u   | v                  |     |     |     |     |     |
PhaseII(Discovery): DefineC ={(u,v)∈V(cid:98) o ld ×V(cid:98) o ld |(u,v)∈/ B}. Addtop
| 15                 |                |     |     | oo-nobank |                |     | t + 1 | t + 1 |     |
| ------------------ | -------------- | --- | --- | --------- | -------------- | --- | ----- | ----- | --- |
| c                  | pairsscoredbyM |     |     |           | toE(cid:98)t+1 | .   |       |       |     |
| (cid:98) oo-nobank |                |     |     | oo-nobank |                |     |       |       |     |
PhaseIII(InductiveGrowth): DefineC =V(cid:98) o ld ×V(cid:98) n ew c pairsscoredbyM
| 16  |     |     |     |     | on  | t + 1 | t + 1 . Addtop | (cid:98) on | on to |
| --- | --- | --- | --- | --- | --- | ----- | -------------- | ----------- | ----- |
E(cid:98)t+1 .
17 PhaseIV(NewCommunity): DefineC ={(u,v)∈V(cid:98) n ew ×V(cid:98) n ew ,u̸=v}. Addtop c pairs
|           |     |                   |     |     | nn  |     | t + 1 | t + 1 | (cid:98) nn |
| --------- | --- | ----------------- | --- | --- | --- | --- | ----- | ----- | ----------- |
| scoredbyM |     | nn toE(cid:98)t+1 | .   |     |     |     |       |       |             |
18 Note: EmbeddingshmayoptionallybeupdatedviaEncbetweenphasestoreflectincremental
structuralchanges.
19 5. EdgeBankMaintenance.
20 UpdateB ←(B∪E(cid:98)t+1 )\E expired whereE expired areedgesolderthanksnapshots.
| returnG(cid:98)t+1 | =(V(cid:98)t+1 |     | ,E(cid:98)t+1 ), | B.  |     |     |     |     |     |
| ------------------ | -------------- | --- | ---------------- | --- | --- | --- | --- | --- | --- |
21
D.5 AdaptingLink-PredictionBaselinestoGraphConstruction
808
Benchmark models are based on their ability to predict edge existence. For each potential edge
candidate, the probability of edge (u,v), the model outputs a probability pt representing the
uv
likelihoodofanedgeappearingattimestampt. Welearnaglobalthresholdforthedatasetandthe
modeltoproduceabinaryadjacencymatrixfromtheprobabilityvalues. Specifically,anedgeis
includedinthepredictedgraphifandonlyif
(cid:40)
ifp(t)
|     |     |     |     |     |        | 1   | ≥T  |     |     |
| --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- |
|     |     |     |     |     | e(t) = |     | uv  |     |     |
|     |     |     |     |     | uv     | 0   |     |     |     |
otherwise
Existingmethodsprimarilyperformlinkpredictionthroughreceivingasetofcandidateedgeswhich
| areasubsetofknownpossibleedgesintheuniverse. |     |     |     |     |      |     | Thatis |     |     |
| -------------------------------------------- | --- | --- | --- | --- | ---- | --- | ------ | --- | --- |
|                                              |     |     |     |     | E    | =E  | ∪E     |     |     |
|                                              |     |     |     |     | cand | pos | neg    |     |     |
22

809 WhereE isthesetof“true”edgesexistinginthecurrentsnapshot(thepositivecandidates)andE
| pos |     |     |     |     |     |     | neg |
| --- | --- | --- | --- | --- | --- | --- | --- |
isasampledsetofnon-existentedges(thenegativecandidates),wheretypically|E |=X·|E |
| 810      |      |     |     |     |     | neg | pos |
| -------- | ---- | --- | --- | --- | --- | --- | --- |
| forsomeX | ∈Z+. |     |     |     |     |     |     |
811
However,forgraphconstruction,wherethesetofknownnodesinG(cid:98)t+1 isunknown,candidateedges
wereselectedasallpossibleedgesbetweenallnodesinthenodeuniverseacrossalltimestepsT.
Thatis
|     |     | E   | ={(u,v,t)|u,v | ∈V,u̸=v,t∈T} |     |     |     |
| --- | --- | --- | ------------- | ------------ | --- | --- | --- |
cand
| Edge selection | and threshold | optimization: |     | .            |                |             |        |
| -------------- | ------------- | ------------- | --- | ------------ | -------------- | ----------- | ------ |
|                |               |               |     | To translate | the continuous | probability | scores |
producedbybaselinemodelsintoadiscreteadjacencystructure,weimplementaglobaldecision
thresholdτ. Weoptimizeτ usingthevalidationpartitionG byperformingagridsearchoverthe
val
interval[0,1]withastepsizeof0.01. Theoptimalthresholdτ∗isselectedtomaximizetheF1-score
acrossallvalidationsnapshots:
τ∗
|     |     |     | =argmaxF1(E(cid:98)τ | ,E    | )   |     |     |
| --- | --- | --- | -------------------- | ----- | --- | --- | --- |
|     |     |     |                      | τ val |     |     |     |
812 whereE(cid:98)τ = {e ∈ E cand | p e ≥ τ}. Thisoptimizedthresholdissubsequentlyappliedtothetest
813 partitiontoreconstructthefinaladjacencymatrices.
BoundedAdjacencyReconstruction. Toensurescalabilityacrossdatasetsofvaryingdensitiesand
topreventmemoryoverflow(OOM)duringtheconstructionphase,weimplementaheuristiccapon
graphgrowth. ForeachpredictedsnapshotG(cid:98)t ,thetotalnumberofedgesisrestrictedtoatmostk ,
t
definedrelativetothecardinalityoftheprecedingsnapshot:
|     |     |     |     | k t =5×|E t−1 | |     |     |     |
| --- | --- | --- | --- | --------------- | --- | --- | --- |
Ifthenumberofcandidateedgesexceedingthedecisionthresholdτ surpassesthislimit,weperform
814
top-kfiltering,selectingonlythek edgeswiththehighestpredictedprobabilities. Thisapproach
| 815 |     |     | t   |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
816 ensures computational tractability while preserving the most structurally significant interactions
817 accordingtothemodel’sconfidencescores.
E StabilityTheorem
818
819 Inthispart,wegivetheproofofProposition4.1. Anaturalquestioniswhetherthefiltration-based
820 descriptorΦ(G )isareliablesnapshotstateforforecasting: iftwosnapshotsdifferbyonlyafew
t
edges, do their descriptors remain close? This is essential for TOPOGED because the topology
821
predictoristrainedtoforecastΦ(G )frompastdescriptors,anditspredictionswillinevitablycarry
| 822 |     |     | t+1 |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
smallerrors. Ifthedescriptorweresensitivetominorstructuralperturbations,theseerrorscould
823
824 cascadeintolargereconstructionmistakesinthedecoder. Thefollowingpropositionshowsthatthe
825 degree-sublevelfiltrationdescriptorisstableinaprecisesense: editingatmostδedgesinasnapshot
perturbsthenode-countsequenceX byatmost2δinℓ norm,andperturbstheinduced-edge-count
| 826 |     |     |     | 1   |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
sequenceY byanamountcontrolledbyδandthefiltrationthresholds. Thisstabilityjustifiesusing
827
Φ(G )asaforecastingtarget:smallpredictionerrorsindescriptorspacecorrespondtosmallstructural
828 t
deviationsinthereconstructedsnapshot,makingtheinversetopologyapproachprincipledratherthan
829
830 heuristic.
Proposition4.1(Stabilityunderedgeedits). LetG = (V,E)andG′ = (V,E′)begraphsonthe
831
samenodesetwith|E△E′| ≤ δ. LetX,Y andX′,Y′ bethedegree-sublevelfiltrationnodeand
832
| induced-edgecountsequencescomputedusingthresholdsϵ |     |     |     |     | <···<ϵ . |     |     |
| -------------------------------------------------- | --- | --- | --- | --- | -------- | --- | --- |
| 833                                                |     |     |     | 1   | n        |     |     |
(i)Ifthethresholdsareconsecutiveintegers{0,1,...,D},then∥X−X′∥
| 834 |     |     |     |     | 1   | ≤2δ. |     |
| --- | --- | --- | --- | --- | --- | ---- | --- |
(cid:80)n
| 835 (ii)Foranythresholds,∥Y |     | −Y′∥ | ≤δ  | (1+2ϵ ). |     |     |     |
| --------------------------- | --- | ---- | --- | -------- | --- | --- | --- |
|                             |     |      | 1   | i=1 i    |     |     |     |
Proof. Let∆d(v)=deg (v)−deg (v). EachedgeinE△E′changesthedegreesofexactlytwo
| 836 |     | G   | G′  |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- |
837 endpointsby±1,hence
(cid:88)
|∆d(v)|≤2|E△E′|≤2δ.
v∈V
Part(i). Assumethresholdsarek ∈ {0,1,...,D}andwritex = |V (G)|whereV (G) = {v :
| 838                        |     |                  |           |     | k k | k        |     |
| -------------------------- | --- | ---------------- | --------- | --- | --- | -------- | --- |
| deg (v)≤k},andsimilarlyx′. |     |                  | Foreachk, |     |     |          |     |
| 839 G                      |     | k                |           |     |     |          |     |
|                            |     | (cid:88)(cid:16) |           |     |     | (cid:17) |     |
−x′
|     | x   | k = | 1{deg | (v)≤k}−1{deg | (v)≤k} | ,   |     |
| --- | --- | --- | ----- | ------------ | ------ | --- | --- |
|     |     | k   |       | G            | G′     |     |     |
v∈V
23

840 sobythetriangleinequality,
|     |     |     |       | (cid:88)(cid:12) |              |     |     |                 | (cid:12) |     |
| --- | --- | --- | ----- | ---------------- | ------------ | --- | --- | --------------- | -------- | --- |
|     |     | |x  | −x′|≤ | (cid:12)1{deg    | (v)≤k}−1{deg |     |     | (v)≤k}(cid:12). |          |     |
|     |     | k   | k     | (cid:12)         | G            |     |     | G′              | (cid:12) |     |
v∈V
Summingoverkandswappingtheorderofsummationgives
841
|     |        |     | (cid:88) D |     | (cid:88)(cid:88) D | (cid:12)      |              |     |                 | (cid:12) |
| --- | ------ | --- | ---------- | --- | ------------------ | ------------- | ------------ | --- | --------------- | -------- |
|     | ∥X−X′∥ |     | −x′|≤      |     |                    | (cid:12)1{deg |              |     | (v)≤k}(cid:12). |          |
|     |        | 1 = | |x k       |     |                    | (cid:12)      | (v)≤k}−1{deg |     | G′              | (cid:12) |
|     |        |     |            | k   |                    |               | G            |     |                 |          |
|     |        |     | k=0        |     | v∈Vk=0             |               |              |     |                 |          |
Forafixedv,theinnersumcountshowmanyintegerthresholdsliestrictlybetweendeg (v)and
| 842 |     |     |     |     |     |     |     |     |     | G   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
843 deg G′ (v),whichisexactly|∆d(v)|wheneverD ≥max{deg (v),deg G′ (v)}andatmost|∆d(v)|
G
| 844 otherwise. | Therefore, |     |     |     |     |     |     |     |     |     |
| -------------- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
(cid:88)
|     |     |     | ∥X−X′∥ |     | ≤   | |∆d(v)|≤2δ. |     |     |     |     |
| --- | --- | --- | ------ | --- | --- | ----------- | --- | --- | --- | --- |
1
v∈V
|     |     |     |     |     | (G)andS′ |     |     | (G′). | E△E′ |     |
| --- | --- | --- | --- | --- | -------- | --- | --- | ----- | ---- | --- |
845 Part(ii). Fixathresholdϵ i andletS = V ϵi = V ϵi LetF = denotetheset
ofeditededgesandC = E ∩E′ thesetofcommonedges. Anyedgeinthesymmetricdifference
846
E(G[S])△E(G′[S′]) is either an edited edge (in F) or a common edge whose membership in the
847
| inducedsubgraphchangesduetoavertexmovingbetweenS |     |     |     |     |     |     | andS′. |     | Thus |     |
| ------------------------------------------------ | --- | --- | --- | --- | --- | --- | ------ | --- | ---- | --- |
848
|     |     |                          |     |     |     | (cid:12)   | (cid:0)       |     | (cid:1)(cid:12) |     |
| --- | --- | ------------------------ | --- | --- | --- | ---------- | ------------- | --- | --------------- | --- |
|     |     | |E(G[S])△E(G′[S′])|≤|F|+ |     |     |     | (cid:12)C∩ | (S×S)△(S′×S′) |     | (cid:12).       |     |
849 Thefirsttermsatisfies|F| ≤ δ. Forthesecondterm,acommonedgechangesmembershipinthe
inducedsubgraphonlyifatleastoneofitsendpointsliesinS△S′,so
850
|     |     |     | (cid:12) (cid:0)         |     |     | (cid:1)(cid:12) | (cid:88) |     |      |     |
| --- | --- | --- | ------------------------ | --- | --- | --------------- | -------- | --- | ---- | --- |
|     |     |     | (cid:12)C∩ (S×S)△(S′×S′) |     |     | (cid:12)≤       |          | deg | (v), |     |
C
v∈S△S′
where deg (v) denotes the degree of v in the common-edge graph (V,C). For any v ∈ S△S′,
| 851 | C   |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
membershipdiffersbetweenS andS′ onlyifdeg (v)̸=deg (v),andeitherv ∈S orv ∈S′,so
| 852     |         |        |          |            |         | G          | G′  |            |     |     |
| ------- | ------- | ------ | -------- | ---------- | ------- | ---------- | --- | ---------- | --- | --- |
| min(deg | (v),deg | (v))≤ϵ | . SinceC | ⊆E         | andC    | ⊆E′,wehave |     |            |     |     |
| 853     | G       | G′     | i        |            |         |            |     |            |     |     |
|         |         |        | deg      | (v)≤mindeg | (cid:0) | (v),deg    | (v) | (cid:1) ≤ϵ | .   |     |
|         |         |        | C        |            |         | G          | G′  |            | i   |     |
(cid:80)
| Moreover,|S△S′|≤ |     |     | |∆d(v)|≤2δ.              |     | Combining, |                 |            |     |     |     |
| ---------------- | --- | --- | ------------------------ | --- | ---------- | --------------- | ---------- | --- | --- | --- |
| 854              |     |     | v∈V                      |     |            |                 |            |     |     |     |
|                  |     |     | (cid:12) (cid:0)         |     |            | (cid:1)(cid:12) |            |     |     |     |
|                  |     |     | (cid:12)C∩ (S×S)△(S′×S′) |     |            | (cid:12)≤ϵ      | |S△S′|≤2δϵ |     | .   |     |
|                  |     |     |                          |     |            |                 | i          |     | i   |     |
855 Therefore,
|                                |     | |y −y′|≤|E(G[S])△E(G′[S′])|≤δ+2δϵ |     |       |     |     |     | =δ(1+2ϵ | ).  |     |
| ------------------------------ | --- | --------------------------------- | --- | ----- | --- | --- | --- | ------- | --- | --- |
|                                |     | i                                 | i   |       |     |     |     | i       | i   |     |
| Summingoveralligivesthestatedℓ |     |                                   |     | bound |     |     |     |         |     |     |
| 856                            |     |                                   |     | 1     |     |     |     |         |     |     |
n
(cid:88)
|     |     |     |     | ∥Y −Y′∥ | ≤δ  | (1+2ϵ |     | ).  |     |     |
| --- | --- | --- | --- | ------- | --- | ----- | --- | --- | --- | --- |
|     |     |     |     |         | 1   |       | i   |     |     |     |
i=1
F DetailedResultsByDataset
857
Inthissection,wepresentagranularperformanceanalysisacrossallevaluateddatasetstocontextual-
858
859 izethemacroaveragesprovidedinTable3. TOPOGEDconsistentlydemonstratessuperiorpredictive
860 capacity,withthemostsignificantgainsoccurringinF1 Nodes,Edge F1,andF1 Old Nodes.
Notably,whilemostbaselinemodelsstruggletomaintainnon-zeroscoresinedge-levelforecasting,
861
TOPOGED maintainsarobustperformancemargin. Furthermore, themodelexhibitssubstantial
862
precisionincapturinggraphtopology,asevidencedbytheStructure Evaluationmetrics. On
863
864 average,TOPOGEDoutperformsthestrongestbaselinein60%ofgraphpropertytasks,frequently
865 achievingresultsordersofmagnitudeclosertothegroundtruththanthenearestcompetingmodel.
Detailed results for all datasets are reported in Tables 11–24, covering CollegeMessage, Math-
866
Overflow,Adex,Aeternity,Aion,Aragon,Bancor,Centra,Cindicator,Coindash,DGD,Iconomi,
867
Reddit_B,andTGBL-Wiki.
868
24

Table11: Comprehensiveevaluationresults—includingedge,node,andstructuralmetrics—forall
benchmark models on CollegeMsg. → 0 indicates that values closer to 0 are better. ↑ denotes
thathighervaluesarebetter,while↓denotesthatlowervaluesarebetter. Boldindicatesbest,the
second-bestisunderlined.
Metric HTGN ROLAND VGRNN TGCN GCLSTM EvolveGCN TopoGED
noitaulavEegdE
oo-bankPrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.26±0.12
oo-bankRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.45±0.21
oo-bankF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.32±0.12
oo-nobankPrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.03
oo-nobankRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.05
oo-nobankF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.04
EdgePrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.16±0.07
EdgeRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.25±0.13
EdgeF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.19±0.07
Numo-nPredicted→0 1.57±2.62 0.54±2.22 −0.16±2.00 1.75±8.89 0.15±2.07 1.15±3.13 0.96±2.42
Numn-nPredicted→0 −0.04±0.19 −0.04±0.19 −0.04±0.19 0.14±0.71 −0.04±0.19 0.09±0.33 0.07±0.60
noitaulavEedoN PrecisionNodes↑ 0.02±0.02 0.02±0.02 0.03±0.04 0.01±0.02 0.02±0.02 0.02±0.02 0.43±0.11
RecallNodes↑ 0.06±0.04 0.02±0.02 0.02±0.02 0.01±0.02 0.01±0.01 0.09±0.05 0.46±0.13
F1Nodes↑ 0.03±0.02 0.02±0.02 0.02±0.03 0.01±0.02 0.01±0.02 0.04±0.02 0.44±0.10
PrecisionOldNodes↑ 0.02±0.02 0.02±0.02 0.03±0.04 0.00±0.01 0.02±0.02 0.02±0.01 0.43±0.11
RecallOldNodes↑ 0.06±0.04 0.02±0.02 0.02±0.03 0.00±0.01 0.01±0.02 0.09±0.05 0.46±0.13
F1OldNodes↑ 0.04±0.02 0.02±0.02 0.02±0.03 0.00±0.01 0.01±0.02 0.04±0.02 0.43±0.10
NewNodesPredicted↑ −0.15±0.64 −0.30±0.71 −0.57±0.67 −0.31±1.58 −0.53±0.72 1.03±2.96 0.42±1.41
noitaulavEerutcurtS
AvgNodeDegree→0 0.37±0.21 2.59±1.19 4.75±1.60 3.18±0.73 4.70±1.69 −0.05±0.10 0.36±0.19
UniqueDegreeCount→0 0.72±0.55 2.20±1.10 1.89±1.03 2.52±0.97 2.44±1.03 0.37±0.47 0.64±0.48
DegreeCentrality→0 −0.45±0.11 3.02±2.58 9.07±3.84 4.20±1.01 9.07±4.30 −0.73±0.07 0.28±0.24
AssortativityCoefficient→0 −5.19±6.47 6.98±17.90 6.02±14.07 5.45±13.81 5.49±13.43 −8.03±12.29 1.13±7.92
ClusteringCoefficient→0 0.39±1.03 1.24±3.92 1.64±4.80 1.33±3.75 1.52±4.12 0.20±0.48 0.07±0.06
Density→0 −0.45±0.11 3.02±2.58 9.07±3.84 4.20±1.01 9.07±4.30 −0.73±0.07 0.28±0.24
NumTriangles→0 22.46±16.69 130.32±100.27 214.75±143.20 147.96±99.07 219.50±165.61 18.43±22.63 3.11±3.20
DescriptorNorm→0 157.41±64.71 91.78±36.40 103.52±45.81 86.98±27.94 100.15±37.70 318.29±120.85 35.42±11.91
MedianExtraNodes→0 57.00±6.14 0.00±1.34 0.00±0.00 0.00±0.00 0.00±0.00 93.50±10.56 0.00±1.19
MedianMissingNodes→0 0.00±0.00 2.50±2.22 15.00±1.34 9.00±0.99 15.00±2.19 0.00±0.00 0.50±1.19
MedianExtraEdges→0 57.00±8.15 63.50±8.08 57.00±8.11 57.00±8.39 57.00±8.39 57.00±8.20 7.50±2.67
MedianMissingEdges→0 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00
Table12: Comprehensiveevaluationresults—includingedge,node,andstructuralmetrics—forall
benchmarkmodelsonMathOverflow. → 0indicatesthatvaluescloserto0arebetter. ↑denotes
thathighervaluesarebetter,while↓denotesthatlowervaluesarebetter. Boldindicatesbest,the
second-bestisunderlined.
Metric HTGN ROLAND VGRNN TGCN GCLSTM EvolveGCN TopoGED
noitaulavEegdE
oo-bankPrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.07±0.03
oo-bankRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.09±0.04
oo-bankF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.08±0.04
oo-nobankPrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.01
oo-nobankRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.01
oo-nobankF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.01
EdgePrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.02±0.01
EdgeRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.03±0.01
EdgeF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.02±0.01
Numo-nPredicted→0 1.67±1.16 1.99±1.45 2.56±1.91 −0.36±0.31 3.57±2.53 2.48±1.69 0.14±0.40
Numn-nPredicted→0 85.49±76.27 82.35±79.78 76.85±68.83 98.11±87.26 69.71±61.45 79.39±75.28 0.45±1.63
noitaulavEedoN PrecisionNodes↑ 0.01±0.00 0.01±0.01 0.01±0.02 0.01±0.01 0.02±0.02 0.01±0.00 0.42±0.06
RecallNodes↑ 0.01±0.01 0.01±0.01 0.00±0.01 0.00±0.01 0.01±0.01 0.02±0.01 0.41±0.04
F1Nodes↑ 0.01±0.01 0.01±0.01 0.01±0.01 0.01±0.01 0.01±0.01 0.01±0.00 0.41±0.04
PrecisionOldNodes↑ 0.05±0.04 0.09±0.07 0.11±0.20 0.18±0.23 0.09±0.08 0.05±0.03 0.36±0.06
RecallOldNodes↑ 0.01±0.01 0.01±0.01 0.01±0.01 0.00±0.01 0.01±0.01 0.02±0.01 0.35±0.04
F1OldNodes↑ 0.02±0.01 0.02±0.01 0.01±0.01 0.01±0.01 0.02±0.01 0.03±0.01 0.35±0.04
NewNodesPredicted↑ 10.94±5.47 4.75±2.16 1.61±1.21 2.61±1.61 2.26±1.64 17.13±8.81 0.07±0.51
noitaulavEerutcurtS
AvgNodeDegree→0 0.73±0.15 2.47±0.80 6.79±2.14 4.92±0.48 4.82±1.08 0.13±0.15 0.07±0.06
UniqueDegreeCount→0 0.80±0.28 2.22±0.57 1.65±0.73 2.72±0.58 2.65±0.63 1.52±0.41 0.46±0.27
DegreeCentrality→0 0.11±0.17 3.79±2.31 23.63±14.41 12.46±2.13 12.11±4.04 −0.51±0.16 0.09±0.17
AssortativityCoefficient→0 −188.57±1073.62 141.27±811.37 144.25±825.39 127.30±740.41 147.56±851.57 −54.55±314.30 51.44±300.12
ClusteringCoefficient→0 13.17±9.39 18.98±14.78 30.79±23.61 29.39±20.61 26.02±18.98 6.50±5.57 2.45±3.26
Density→0 0.11±0.17 3.79±2.31 23.63±14.41 12.46±2.13 12.11±4.04 −0.51±0.16 0.09±0.17
NumTriangles→0 174.69±103.80 418.85±271.60 972.69±616.75 646.03±386.53 707.80±425.57 209.57±153.77 6.45±6.70
DescriptorNorm→0 349.89±72.70 287.23±48.77 378.51±103.85 330.13±45.08 329.96±49.60 626.17±167.32 98.76±25.30
MedianExtraNodes→0 80.00±4.28 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 206.00±14.61 0.00±1.87
MedianMissingNodes→0 0.00±0.00 25.00±6.56 91.00±7.21 78.00±4.21 80.00±4.70 0.00±0.00 3.00±3.53
MedianExtraEdges→0 230.00±14.01 209.00±23.18 227.00±16.99 227.00±15.30 227.00±16.16 227.00±16.05 8.00±4.38
MedianMissingEdges→0 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±2.34
25

Table13: Comprehensiveevaluationresults—includingedge,node,andstructuralmetrics—forall
benchmarkmodelsonAdex. →0indicatesthatvaluescloserto0arebetter. ↑denotesthathigher
valuesarebetter,while↓denotesthatlowervaluesarebetter. Boldindicatesbest,thesecond-bestis
underlined.
Metric HTGN ROLAND VGRNN TGCN GCLSTM EvolveGCN TopoGED
noitaulavEegdE
oo-bankPrecision↑ 0.01±0.05 0.00±0.00 0.09±0.21 0.00±0.00 0.25±0.23 0.03±0.15 0.20±0.12
oo-bankRecall↑ 0.00±0.00 0.00±0.00 0.01±0.03 0.00±0.00 0.06±0.06 0.00±0.02 0.27±0.13
oo-bankF1↑ 0.00±0.01 0.00±0.00 0.02±0.05 0.00±0.00 0.09±0.08 0.00±0.03 0.22±0.10
oo-nobankPrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.03
oo-nobankRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.02 0.00±0.00 0.01±0.03
oo-nobankF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.03
EdgePrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.10±0.05
EdgeRecall↑ 0.00±0.00 0.00±0.00 0.00±0.01 0.00±0.00 0.02±0.02 0.00±0.01 0.09±0.05
EdgeF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.00 0.00±0.00 0.09±0.04
Numo-nPredicted→0 −0.88±0.10 3.41±1.97 2.55±2.19 −0.91±0.10 4.05±3.35 0.29±2.92 0.19±0.82
Numn-nPredicted→0 −0.50±0.68 57.73±45.97 43.89±45.76 −0.89±0.32 24.11±23.73 21.25±52.82 0.62±1.69
noitaulavEedoN PrecisionNodes↑ 0.00±0.00 0.04±0.03 0.05±0.03 0.00±0.00 0.08±0.02 0.01±0.01 0.55±0.12
RecallNodes↑ 0.00±0.01 0.02±0.01 0.02±0.01 0.00±0.00 0.10±0.03 0.03±0.05 0.56±0.11
F1Nodes↑ 0.00±0.00 0.03±0.01 0.03±0.01 0.00±0.00 0.09±0.02 0.01±0.02 0.54±0.09
PrecisionOldNodes↑ 0.00±0.00 0.07±0.03 0.10±0.07 0.00±0.00 0.14±0.04 0.01±0.03 0.31±0.09
RecallOldNodes↑ 0.00±0.01 0.03±0.02 0.04±0.02 0.00±0.00 0.17±0.05 0.04±0.07 0.32±0.07
F1OldNodes↑ 0.00±0.00 0.04±0.02 0.05±0.02 0.00±0.00 0.15±0.03 0.02±0.04 0.30±0.05
NewNodesPredicted↑ −0.85±0.12 −0.33±0.35 −0.48±0.29 −0.98±0.01 0.41±0.46 0.08±1.82 0.12±0.59
noitaulavEerutcurtS
AvgNodeDegree→0 −0.40±0.08 7.91±2.93 7.59±3.32 4.46±1.44 3.26±3.56 −0.12±0.39 0.02±0.15
UniqueDegreeCount→0 −0.60±0.11 1.11±1.04 0.61±0.56 1.87±0.78 0.83±0.95 −0.06±0.65 0.09±0.28
DegreeCentrality→0 −0.85±0.05 21.15±22.28 21.76±14.62 9.77±3.67 3.10±5.80 −0.71±0.15 0.05±0.43
AssortativityCoefficient→0 −2.58±0.89 1.29±0.93 0.99±0.67 1.19±0.60 1.30±0.67 −1.33±1.27 0.88±0.56
ClusteringCoefficient→0 0.37±1.45 38.29±72.55 34.53±65.81 35.92±67.31 31.98±61.46 6.35±13.46 1.54±4.16
Density→0 −0.85±0.05 21.15±22.28 21.76±14.62 9.77±3.67 3.10±5.80 −0.71±0.15 0.05±0.43
NumTriangles→0 2.92±11.49 2182.74±6414.98 1277.94±1705.64 755.62±1227.33 895.09±2437.47 99.45±498.74 5.06±7.96
DescriptorNorm→0 1479.50±1422.56 501.81±485.32 470.69±486.77 351.54±442.18 495.11±627.77 754.83±441.84 180.45±279.10
MedianExtraNodes→0 314.00±32.35 0.00±0.00 0.00±0.00 0.00±0.00 36.00±9.65 211.00±14.68 4.00±3.93
MedianMissingNodes→0 0.00±0.00 39.00±5.65 51.00±6.32 41.00±6.52 0.00±0.00 0.00±0.00 0.00±1.91
MedianExtraEdges→0 132.00±14.89 294.00±25.96 203.00±19.02 148.00±19.28 308.00±29.51 129.00±16.17 1.00±4.89
MedianMissingEdges→0 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±2.50
Table 14: Comprehensive evaluation results—including edge, node, and structural metrics—for
all benchmark models on Aeternity. → 0 indicates that values closer to 0 are better. ↑ denotes
thathighervaluesarebetter,while↓denotesthatlowervaluesarebetter. Boldindicatesbest,the
second-bestisunderlined.
Metric HTGN ROLAND VGRNN TGCN GCLSTM EvolveGCN TopoGED
noitaulavEegdE
oo-bankPrecision↑ 0.00±0.00 0.08±0.28 0.03±0.17 0.00±0.00 0.06±0.23 0.06±0.17 0.20±0.07
oo-bankRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.01 0.28±0.13
oo-bankF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.01 0.22±0.07
oo-nobankPrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.02 0.01±0.02
oo-nobankRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.03
oo-nobankF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.02
EdgePrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.11±0.04
EdgeRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.11±0.05
EdgeF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.11±0.04
Numo-nPredicted→0 −0.81±0.12 −0.10±0.44 0.68±1.02 −0.02±2.52 0.84±1.47 0.31±2.05 0.00±0.33
Numn-nPredicted→0 128.35±75.97 105.26±58.75 196.59±115.27 114.37±83.36 214.64±131.71 233.41±141.90 0.31±1.06
noitaulavEedoN PrecisionNodes↑ 0.00±0.00 0.01±0.01 0.01±0.01 0.00±0.00 0.01±0.00 0.01±0.00 0.48±0.11
RecallNodes↑ 0.00±0.00 0.01±0.01 0.00±0.00 0.00±0.00 0.01±0.01 0.02±0.01 0.47±0.14
F1Nodes↑ 0.00±0.00 0.01±0.01 0.01±0.00 0.00±0.00 0.01±0.00 0.01±0.00 0.46±0.11
PrecisionOldNodes↑ 0.06±0.05 0.29±0.17 0.37±0.33 0.02±0.06 0.34±0.11 0.41±0.12 0.30±0.09
RecallOldNodes↑ 0.01±0.01 0.01±0.01 0.01±0.00 0.00±0.00 0.02±0.01 0.03±0.02 0.29±0.10
F1OldNodes↑ 0.01±0.01 0.03±0.02 0.01±0.01 0.00±0.01 0.03±0.02 0.05±0.03 0.29±0.08
NewNodesPredicted↑ 11.32±4.69 1.09±0.93 0.59±0.73 2.29±2.83 3.72±1.71 6.18±3.06 0.00±0.32
noitaulavEerutcurtS
AvgNodeDegree→0 −0.26±0.43 2.94±1.26 8.71±3.73 2.68±1.89 2.75±2.18 2.08±3.19 0.00±0.08
UniqueDegreeCount→0 −0.33±0.38 1.27±0.58 0.64±0.56 1.23±0.97 1.41±1.21 −0.14±0.85 −0.06±0.24
DegreeCentrality→0 −0.69±0.50 6.50±4.77 30.12±56.24 5.55±4.72 3.52±7.69 3.44±11.12 0.10±0.44
AssortativityCoefficient→0 −2.42±0.34 0.74±0.42 1.10±0.36 0.80±0.45 0.71±0.50 1.46±0.53 0.64±0.31
ClusteringCoefficient→0 8.37±20.58 87.08±195.38 105.80±206.62 51.98±99.18 88.01±196.81 87.57±194.23 6.70±14.43
Density→0 −0.69±0.50 6.50±4.77 30.12±56.24 5.55±4.72 3.52±7.69 3.44±11.12 0.10±0.44
NumTriangles→0 112.86±289.18 1235.52±1708.21 4023.98±7267.32 832.73±871.23 1655.07±3540.89 1395.16±4023.10 10.21±14.78
DescriptorNorm→0 2185.37±1115.69 666.41±657.85 1289.74±1286.05 754.89±728.87 1272.95±1046.26 1567.32±1128.84 384.54±599.03
MedianExtraNodes→0 625.00±25.84 0.00±0.00 0.00±0.00 0.00±4.27 138.00±16.87 322.50±19.87 1.00±5.34
MedianMissingNodes→0 0.00±0.00 98.00±22.37 106.00±11.25 58.50±10.87 0.00±0.44 0.00±0.00 2.00±12.34
MedianExtraEdges→0 325.50±33.60 309.00±29.77 758.00±65.10 318.50±31.70 817.50±76.77 859.50±86.60 1.50±7.05
MedianMissingEdges→0 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±15.72
26

Table15: Comprehensiveevaluationresults—includingedge,node,andstructuralmetrics—forall
benchmarkmodelsonAion. →0indicatesthatvaluescloserto0arebetter. ↑denotesthathigher
valuesarebetter,while↓denotesthatlowervaluesarebetter. Boldindicatesbest,thesecond-bestis
underlined.
Metric HTGN ROLAND VGRNN TGCN GCLSTM EvolveGCN TopoGED
noitaulavEegdE
oo-bankPrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.13±0.35 0.00±0.00 0.07±0.03
oo-bankRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.01 0.00±0.00 0.16±0.08
oo-bankF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.01 0.00±0.00 0.09±0.04
oo-nobankPrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.02±0.09 0.00±0.00 0.03±0.03
oo-nobankRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.02
oo-nobankF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.02±0.02
EdgePrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.15±0.05
EdgeRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.15±0.06
EdgeF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.15±0.05
Numo-nPredicted→0 −0.74±0.07 −0.79±0.07 0.52±0.77 −0.84±0.11 −0.17±0.22 −0.47±0.36 0.02±0.22
Numn-nPredicted→0 202.57±177.05 206.59±200.54 174.04±157.08 260.36±206.20 176.97±153.41 196.50±172.56 1.05±3.46
noitaulavEedoN PrecisionNodes↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.02±0.01 0.01±0.02 0.45±0.06
RecallNodes↑ 0.00±0.00 0.01±0.00 0.00±0.00 0.00±0.00 0.01±0.00 0.00±0.00 0.45±0.07
F1Nodes↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.00 0.00±0.01 0.45±0.05
PrecisionOldNodes↑ 0.01±0.02 0.09±0.03 0.08±0.08 0.00±0.01 0.35±0.11 0.18±0.21 0.14±0.04
RecallOldNodes↑ 0.00±0.00 0.01±0.01 0.00±0.00 0.00±0.00 0.01±0.00 0.00±0.01 0.13±0.04
F1OldNodes↑ 0.00±0.00 0.02±0.01 0.01±0.01 0.00±0.00 0.02±0.01 0.01±0.01 0.13±0.04
NewNodesPredicted↑ 5.77±1.39 4.28±1.34 0.11±0.36 5.03±3.45 −0.33±0.13 −0.50±0.11 0.01±0.21
noitaulavEerutcurtS
AvgNodeDegree→0 −0.09±0.10 0.17±0.12 5.35±1.64 1.33±2.48 7.78±0.98 11.41±2.50 0.01±0.06
UniqueDegreeCount→0 0.02±0.20 0.08±0.45 0.88±0.61 0.46±1.36 3.11±0.84 2.69±0.81 −0.17±0.21
DegreeCentrality→0 −0.66±0.08 −0.42±0.21 14.79±9.11 3.16±8.66 31.15±7.68 63.10±25.23 0.05±0.22
AssortativityCoefficient→0 −2.60±0.44 1.12±0.53 0.87±0.46 1.27±0.83 0.19±0.38 −0.05±0.31 1.08±0.37
ClusteringCoefficient→0 10.20±11.05 5.57±11.05 76.65±82.51 12.08±30.30 59.78±63.33 66.35±71.37 7.38±10.13
Density→0 −0.66±0.08 −0.42±0.21 14.79±9.11 3.16±8.66 31.15±7.68 63.10±25.23 0.05±0.22
NumTriangles→0 87.45±65.09 48.10±66.90 1912.40±1276.79 490.73±767.46 2594.00±1728.90 3245.35±2425.08 15.53±16.14
DescriptorNorm→0 1330.29±301.01 1164.38±502.01 747.93±226.62 1902.54±1532.02 790.99±218.30 909.13±280.69 279.83±115.95
MedianExtraNodes→0 475.50±19.30 321.50±22.01 0.00±0.00 353.50±104.50 0.00±0.00 0.00±0.00 0.00±3.01
MedianMissingNodes→0 0.00±0.00 0.00±0.00 146.50±12.55 0.00±0.00 184.50±22.56 200.00±28.53 8.50±6.23
MedianExtraEdges→0 379.50±25.05 363.50±26.38 409.00±57.09 465.00±65.47 378.50±24.70 386.00±26.84 2.50±5.93
MedianMissingEdges→0 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 4.00±6.50
Table16: Comprehensiveevaluationresults—includingedge,node,andstructuralmetrics—forall
benchmarkmodelsonAragon. →0indicatesthatvaluescloserto0arebetter. ↑denotesthathigher
valuesarebetter,while↓denotesthatlowervaluesarebetter. Boldindicatesbest,thesecond-bestis
underlined. OOM indicateout-of-memory.
Metric HTGN ROLAND VGRNN TGCN GCLSTM EvolveGCN TopoGED
noitaulavEegdE
oo-bankPrecision↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.07±0.06
oo-bankRecall↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.13±0.14
oo-bankF1↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.08±0.08
oo-nobankPrecision↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.01
oo-nobankRecall↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.01
oo-nobankF1↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.01
EdgePrecision↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.22±0.08
EdgeRecall↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.23±0.09
EdgeF1↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.22±0.08
Numo-nPredicted→0 −0.63±0.10 −0.87±0.04 OOM −0.90±0.04 −0.78±0.20 −0.77±0.33 0.07±0.29
Numn-nPredicted→0 157.99±97.18 173.19±108.70 OOM 171.62±104.11 335.32±196.71 168.11±109.39 0.58±1.81
noitaulavEedoN PrecisionNodes↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.01±0.00 0.00±0.00 0.51±0.06
RecallNodes↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.01±0.01 0.00±0.01 0.53±0.08
F1Nodes↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.01±0.00 0.00±0.00 0.51±0.04
PrecisionOldNodes↑ 0.00±0.01 0.01±0.03 OOM 0.00±0.00 0.28±0.10 0.06±0.09 0.16±0.04
RecallOldNodes↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.02±0.01 0.01±0.02 0.16±0.05
F1OldNodes↑ 0.00±0.00 0.00±0.01 OOM 0.00±0.00 0.04±0.02 0.01±0.02 0.16±0.04
NewNodesPredicted↑ 8.14±2.58 3.33±1.38 OOM 0.08±0.44 2.53±1.10 3.36±1.30 0.06±0.27
noitaulavEerutcurtS
AvgNodeDegree→0 −0.44±0.03 0.26±0.28 OOM 4.21±1.17 2.14±1.14 0.25±0.23 0.01±0.06
UniqueDegreeCount→0 −0.79±0.11 −0.50±0.24 OOM 2.53±0.95 0.45±0.70 1.07±0.69 0.13±0.24
DegreeCentrality→0 −0.87±0.04 −0.31±0.47 OOM 10.29±4.69 1.09±1.33 −0.37±0.28 0.01±0.27
AssortativityCoefficient→0 −2.64±1.47 2.08±0.47 OOM 0.90±0.35 1.84±0.44 −0.07±1.11 0.79±0.33
ClusteringCoefficient→0 −0.82±0.38 −0.83±0.38 OOM 23.32±19.28 29.93±23.87 4.42±5.72 1.01±2.22
Density→0 −0.87±0.04 −0.31±0.47 OOM 10.29±4.69 1.09±1.33 −0.37±0.28 0.01±0.27
NumTriangles→0 −0.75±0.57 −0.83±0.38 OOM 595.75±473.80 463.09±516.32 70.68±74.09 2.62±3.67
DescriptorNorm→0 1676.71±786.88 523.55±295.90 OOM 296.62±130.53 516.47±178.13 419.98±232.31 127.58±86.93
MedianExtraNodes→0 377.50±24.71 117.50±10.36 OOM 0.00±0.00 84.50±6.86 120.50±10.29 8.50±4.56
MedianMissingNodes→0 0.00±0.00 0.00±0.00 OOM 55.00±6.53 0.00±0.00 0.00±0.00 0.00±1.05
MedianExtraEdges→0 146.00±12.86 152.50±11.25 OOM 145.50±13.10 377.50±26.38 145.50±13.57 9.00±4.31
MedianMissingEdges→0 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.91
27

Table17: Comprehensiveevaluationresults—includingedge,node,andstructuralmetrics—forall
benchmarkmodelsonBancor. →0indicatesthatvaluescloserto0arebetter. ↑denotesthathigher
valuesarebetter,while↓denotesthatlowervaluesarebetter. Boldindicatesbest,thesecond-bestis
underlined. OOM indicateout-of-memory.
Metric HTGN ROLAND VGRNN TGCN GCLSTM EvolveGCN TopoGED
noitaulavEegdE
oo-bankPrecision↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.23±0.42 0.01±0.05 0.25±0.11
oo-bankRecall↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.01±0.01 0.00±0.00 0.34±0.15
oo-bankF1↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.01±0.03 0.00±0.00 0.27±0.10
oo-nobankPrecision↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.01±0.05 0.00±0.00 0.01±0.03
oo-nobankRecall↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.02
oo-nobankF1↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.01 0.00±0.00 0.01±0.03
EdgePrecision↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.15±0.08
EdgeRecall↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.14±0.07
EdgeF1↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.14±0.07
Numo-nPredicted→0 −0.45±0.22 −0.29±0.45 OOM 0.34±0.53 0.59±0.51 −0.11±0.40 0.04±0.35
Numn-nPredicted→0 69.62±65.14 64.20±59.69 OOM 55.66±52.53 107.34±97.87 64.87±61.32 0.56±1.80
noitaulavEedoN PrecisionNodes↑ 0.00±0.00 0.01±0.01 OOM 0.00±0.00 0.03±0.01 0.03±0.02 0.58±0.09
RecallNodes↑ 0.00±0.00 0.01±0.00 OOM 0.00±0.00 0.02±0.01 0.01±0.01 0.57±0.10
F1Nodes↑ 0.00±0.00 0.01±0.00 OOM 0.00±0.00 0.03±0.01 0.02±0.01 0.57±0.07
PrecisionOldNodes↑ 0.00±0.00 0.62±0.44 OOM 0.01±0.03 0.51±0.10 0.34±0.16 0.33±0.09
RecallOldNodes↑ 0.00±0.00 0.01±0.01 OOM 0.00±0.00 0.04±0.01 0.03±0.01 0.32±0.08
F1OldNodes↑ 0.00±0.00 0.02±0.01 OOM 0.00±0.01 0.07±0.02 0.05±0.03 0.32±0.07
NewNodesPredicted↑ 8.58±3.85 0.48±0.61 OOM −0.01±0.33 0.41±0.50 0.27±0.47 0.03±0.31
noitaulavEerutcurtS
AvgNodeDegree→0 −0.39±0.05 3.18±1.22 OOM 4.44±0.87 6.20±1.73 3.69±1.31 0.01±0.09
UniqueDegreeCount→0 −0.63±0.07 1.31±0.59 OOM 1.83±0.66 1.72±0.92 1.63±0.65 0.16±0.24
DegreeCentrality→0 −0.84±0.07 7.45±6.13 OOM 11.67±6.18 12.28±9.08 8.53±6.83 0.08±0.46
AssortativityCoefficient→0 −2.25±0.42 1.39±0.57 OOM 0.94±0.42 1.81±0.70 −2.27±0.74 1.13±0.52
ClusteringCoefficient→0 −0.84±0.34 34.97±42.92 OOM 40.36±48.15 44.31±50.13 29.82±35.45 2.63±4.87
Density→0 −0.84±0.07 7.45±6.13 OOM 11.67±6.18 12.28±9.08 8.53±6.83 0.08±0.46
NumTriangles→0 −0.74±0.58 374.60±295.06 OOM 492.84±468.77 1027.56±1080.19 559.73±549.70 5.15±6.79
DescriptorNorm→0 1656.96±1283.31 319.63±255.20 OOM 365.52±308.79 500.71±299.54 374.58±338.75 171.73±230.58
MedianExtraNodes→0 380.00±20.37 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.00±2.21
MedianMissingNodes→0 0.00±0.00 47.50±5.96 OOM 68.00±4.53 45.50±4.88 56.00±4.12 1.50±1.74
MedianExtraEdges→0 154.00±9.87 158.00±13.16 OOM 153.00±10.12 356.50±11.58 160.00±9.92 1.50±2.91
MedianMissingEdges→0 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.00±1.76
Table18: Comprehensiveevaluationresults—includingedge,node,andstructuralmetrics—forall
benchmarkmodelsonCentra. →0indicatesthatvaluescloserto0arebetter. ↑denotesthathigher
valuesarebetter,while↓denotesthatlowervaluesarebetter. Boldindicatesbest,thesecond-bestis
underlined.
Metric HTGN ROLAND VGRNN TGCN GCLSTM EvolveGCN TopoGED
noitaulavEegdE
oo-bankPrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.12±0.22 0.05±0.22 0.09±0.09
oo-bankRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.02 0.00±0.00 0.18±0.21
oo-bankF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.02±0.04 0.00±0.01 0.11±0.11
oo-nobankPrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.02
oo-nobankRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.02
oo-nobankF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.01
EdgePrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.07±0.06
EdgeRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.08±0.07
EdgeF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.07±0.06
Numo-nPredicted→0 −0.19±0.64 0.05±1.48 0.64±2.27 −0.87±0.18 0.52±1.26 −0.26±0.66 0.46±0.92
Numn-nPredicted→0 419.16±554.55 500.92±743.93 544.65±652.79 153.58±101.27 473.27±499.76 423.32±569.51 0.74±1.74
noitaulavEedoN PrecisionNodes↑ 0.00±0.00 0.00±0.00 0.01±0.01 0.00±0.00 0.02±0.01 0.01±0.01 0.47±0.15
RecallNodes↑ 0.00±0.00 0.01±0.01 0.01±0.01 0.00±0.00 0.01±0.01 0.01±0.01 0.56±0.22
F1Nodes↑ 0.00±0.00 0.01±0.00 0.01±0.01 0.00±0.00 0.01±0.01 0.01±0.01 0.47±0.14
PrecisionOldNodes↑ 0.00±0.00 0.05±0.04 0.23±0.33 0.00±0.00 0.17±0.08 0.22±0.29 0.23±0.13
RecallOldNodes↑ 0.00±0.01 0.02±0.02 0.01±0.01 0.00±0.00 0.03±0.02 0.02±0.02 0.26±0.13
F1OldNodes↑ 0.00±0.00 0.02±0.02 0.02±0.02 0.00±0.00 0.05±0.03 0.03±0.04 0.21±0.11
NewNodesPredicted↑ 10.35±6.75 5.31±6.29 1.75±5.46 2.89±2.64 1.31±2.09 0.96±3.44 0.35±0.76
noitaulavEerutcurtS
AvgNodeDegree→0 −0.39±0.17 0.30±0.55 6.05±5.77 1.49±1.64 4.89±2.68 4.58±3.51 0.03±0.12
UniqueDegreeCount→0 −0.46±0.48 0.43±1.40 0.53±0.74 −0.05±0.73 2.15±1.72 2.14±1.23 0.06±0.30
DegreeCentrality→0 −0.79±0.30 0.52±5.05 21.35±45.22 3.04±4.25 20.43±39.18 24.73±56.03 0.56±3.55
AssortativityCoefficient→0 −1.58±0.61 0.25±2.47 0.26±1.95 0.46±3.29 0.07±2.20 −0.94±0.63 −0.04±1.53
ClusteringCoefficient→0 4.89±18.04 14.82±63.29 56.31±135.01 6.74±23.95 46.57±103.10 35.26±83.73 2.13±4.24
Density→0 −0.79±0.30 0.52±5.05 21.35±45.22 3.04±4.25 20.43±39.18 24.73±56.03 0.56±3.55
NumTriangles→0 138.66±722.69 486.66±1408.43 1843.29±4451.19 126.88±152.34 2946.97±8072.72 1808.73±3466.10 16.99±43.03
DescriptorNorm→0 3578.26±4205.39 1862.97±2565.60 2326.09±3778.83 650.07±1494.33 1758.78±2467.73 1995.73±3215.58 1042.17±1701.40
MedianExtraNodes→0 411.00±73.46 168.00±34.40 0.00±1.30 0.00±31.00 0.00±1.77 0.00±0.00 19.50±11.98
MedianMissingNodes→0 0.00±0.00 0.00±0.00 49.50±29.18 1.00±7.40 21.50±23.48 58.00±10.03 0.00±1.68
MedianExtraEdges→0 154.50±40.46 186.50±53.45 361.50±66.36 115.00±18.19 366.50±49.04 151.50±41.55 26.50±19.25
MedianMissingEdges→0 0.00±0.00 0.00±0.00 0.00±0.00 0.00±2.12 0.00±0.00 0.00±0.00 0.00±1.06
28

Table 19: Comprehensive evaluation results—including edge, node, and structural metrics—for
allbenchmarkmodelsonCindicator. → 0indicatesthatvaluescloserto0arebetter. ↑denotes
thathighervaluesarebetter,while↓denotesthatlowervaluesarebetter. Boldindicatesbest,the
second-bestisunderlined.
Metric HTGN ROLAND VGRNN TGCN GCLSTM EvolveGCN TopoGED
noitaulavEegdE
oo-bankPrecision↑ 0.00±0.00 0.15±0.36 0.06±0.24 0.00±0.00 0.15±0.36 0.00±0.00 0.11±0.07
oo-bankRecall↑ 0.00±0.00 0.00±0.01 0.00±0.01 0.00±0.00 0.00±0.01 0.00±0.00 0.18±0.11
oo-bankF1↑ 0.00±0.00 0.01±0.02 0.00±0.01 0.00±0.00 0.01±0.02 0.00±0.00 0.13±0.07
oo-nobankPrecision↑ 0.00±0.00 0.00±0.01 0.00±0.00 0.00±0.00 0.01±0.06 0.00±0.00 0.02±0.03
oo-nobankRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.02
oo-nobankF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.01 0.00±0.00 0.01±0.02
EdgePrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.12±0.07
EdgeRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.13±0.07
EdgeF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.13±0.07
Numo-nPredicted→0 −0.70±0.08 3.67±1.44 −0.51±0.42 −1.00±0.00 0.22±0.25 −0.70±0.17 0.05±0.26
Numn-nPredicted→0 278.15±150.62 11.05±11.11 268.33±146.33 293.46±159.21 228.10±122.24 276.82±154.10 0.52±1.58
noitaulavEedoN PrecisionNodes↑ 0.00±0.00 0.00±0.00 0.01±0.01 0.00±0.00 0.03±0.01 0.01±0.00 0.51±0.06
RecallNodes↑ 0.00±0.00 0.01±0.00 0.01±0.01 0.00±0.00 0.01±0.00 0.01±0.01 0.52±0.06
F1Nodes↑ 0.00±0.00 0.01±0.00 0.01±0.01 0.00±0.00 0.02±0.01 0.01±0.00 0.51±0.04
PrecisionOldNodes↑ 0.00±0.00 0.17±0.10 0.39±0.33 0.00±0.00 0.55±0.16 0.20±0.09 0.20±0.05
RecallOldNodes↑ 0.00±0.00 0.02±0.01 0.01±0.01 0.00±0.00 0.02±0.01 0.02±0.01 0.20±0.04
F1OldNodes↑ 0.00±0.00 0.03±0.01 0.02±0.02 0.00±0.00 0.04±0.01 0.04±0.02 0.20±0.04
NewNodesPredicted↑ 8.54±1.90 4.60±1.63 0.00±0.41 0.22±0.22 −0.21±0.14 3.82±1.35 0.04±0.23
noitaulavEerutcurtS
AvgNodeDegree→0 −0.39±0.03 0.05±0.07 5.81±3.09 3.91±0.55 6.43±0.52 0.24±0.27 0.00±0.06
UniqueDegreeCount→0 −0.45±0.11 −0.06±0.24 0.45±0.56 2.33±0.70 2.30±0.66 1.18±0.55 0.17±0.21
DegreeCentrality→0 −0.85±0.03 −0.54±0.14 21.57±22.64 8.64±1.77 20.63±3.97 −0.35±0.35 −0.00±0.19
AssortativityCoefficient→0 −1.95±0.26 1.03±0.31 0.33±0.49 0.45±0.23 0.53±0.24 −0.13±0.35 0.32±0.23
ClusteringCoefficient→0 6.44±22.00 2.41±9.06 68.92±217.82 64.29±184.65 66.34±198.51 5.82±25.51 3.44±11.07
Density→0 −0.85±0.03 −0.54±0.14 21.57±22.64 8.64±1.77 20.63±3.97 −0.35±0.35 −0.00±0.19
NumTriangles→0 16.02±15.07 18.88±17.75 1287.65±1062.84 911.88±726.83 1130.89±770.31 74.73±242.41 7.44±8.97
DescriptorNorm→0 1599.32±396.49 848.61±387.64 503.62±232.09 379.57±100.96 437.60±89.09 555.81±203.93 133.74±56.00
MedianExtraNodes→0 490.50±18.93 213.50±22.82 0.00±0.00 0.00±0.00 0.00±0.00 156.00±8.67 8.00±6.02
MedianMissingNodes→0 0.00±0.00 0.00±0.00 93.00±10.35 74.00±5.71 103.00±6.58 0.00±0.00 0.00±3.62
MedianExtraEdges→0 226.50±17.21 238.50±20.94 236.50±17.67 223.00±17.48 234.50±15.77 223.00±18.10 4.00±6.47
MedianMissingEdges→0 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±5.21
Table 20: Comprehensive evaluation results—including edge, node, and structural metrics—for
all benchmark models on Coindash. → 0 indicates that values closer to 0 are better. ↑ denotes
thathighervaluesarebetter,while↓denotesthatlowervaluesarebetter. Boldindicatesbest,the
second-bestisunderlined.
Metric HTGN ROLAND VGRNN TGCN GCLSTM EvolveGCN TopoGED
noitaulavEegdE
oo-bankPrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.05±0.15 0.00±0.00 0.20±0.10
oo-bankRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.01 0.00±0.00 0.28±0.14
oo-bankF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.02 0.00±0.00 0.22±0.10
oo-nobankPrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.02±0.05
oo-nobankRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.04
oo-nobankF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.02±0.04
EdgePrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.13±0.06
EdgeRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.13±0.07
EdgeF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.13±0.06
Numo-nPredicted→0 0.19±0.78 −0.41±0.32 −0.34±1.58 −0.93±0.40 0.58±2.12 −0.28±0.69 0.13±0.71
Numn-nPredicted→0 179.27±383.82 187.30±95.05 240.66±413.80 203.72±402.28 299.16±368.46 212.74±530.69 0.26±1.29
noitaulavEedoN PrecisionNodes↑ 0.00±0.00 0.01±0.01 0.01±0.02 0.00±0.00 0.03±0.01 0.02±0.02 0.49±0.13
RecallNodes↑ 0.00±0.00 0.02±0.02 0.01±0.01 0.00±0.00 0.04±0.02 0.01±0.01 0.49±0.14
F1Nodes↑ 0.00±0.00 0.01±0.01 0.01±0.01 0.00±0.00 0.03±0.01 0.01±0.01 0.47±0.12
PrecisionOldNodes↑ 0.00±0.00 0.13±0.09 0.28±0.43 0.00±0.00 0.77±0.31 0.35±0.40 0.35±0.12
RecallOldNodes↑ 0.00±0.00 0.03±0.03 0.01±0.01 0.00±0.00 0.05±0.03 0.02±0.02 0.34±0.12
F1OldNodes↑ 0.00±0.00 0.05±0.04 0.02±0.03 0.00±0.00 0.10±0.05 0.03±0.03 0.33±0.11
NewNodesPredicted↑ 17.20±13.86 10.08±6.57 1.25±2.07 1.49±1.20 4.21±2.90 3.06±12.39 0.09±0.63
noitaulavEerutcurtS
AvgNodeDegree→0 −0.47±0.06 0.18±0.17 5.83±2.78 3.19±1.83 3.25±3.14 3.32±1.33 0.01±0.10
UniqueDegreeCount→0 −0.75±0.28 −0.14±0.35 0.36±0.46 1.64±0.82 0.39±0.91 1.49±0.76 0.16±0.29
DegreeCentrality→0 −0.87±0.09 −0.44±0.76 18.00±24.20 8.09±16.17 5.51±17.65 9.74±17.48 0.21±1.07
AssortativityCoefficient→0 −1.95±0.94 1.36±0.83 0.72±0.87 0.14±0.52 1.47±0.91 0.28±0.50 0.47±0.57
ClusteringCoefficient→0 0.29±1.68 3.31±12.00 29.96±117.24 25.97±101.14 29.59±114.28 22.53±95.11 3.39±14.63
Density→0 −0.87±0.09 −0.44±0.76 18.00±24.20 8.09±16.17 5.51±17.65 9.74±17.48 0.21±1.07
NumTriangles→0 4.49±25.54 26.28±25.71 830.21±976.67 486.47±1005.20 703.41±1325.11 368.13±281.53 7.10±8.94
DescriptorNorm→0 1216.42±1642.06 499.46±426.32 405.05±624.87 308.18±658.46 426.25±632.52 488.05±1625.19 186.88±532.55
MedianExtraNodes→0 235.50±26.04 114.50±8.74 0.00±0.00 0.00±0.00 30.00±4.15 0.00±0.00 0.50±1.95
MedianMissingNodes→0 0.00±0.00 0.00±0.00 25.50±5.60 21.00±3.98 0.00±0.00 24.50±4.13 0.00±1.79
MedianExtraEdges→0 86.00±10.55 140.00±10.59 151.00±21.81 85.00±9.91 227.00±24.04 85.50±9.77 1.50±2.61
MedianMissingEdges→0 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±1.15
29

Table21: Comprehensiveevaluationresults—includingedge,node,andstructuralmetrics—forall
benchmarkmodelsonDGD.→0indicatesthatvaluescloserto0arebetter. ↑denotesthathigher
valuesarebetter,while↓denotesthatlowervaluesarebetter. Boldindicatesbest,thesecond-bestis
underlined. OOM indicateout-of-memory.
Metric HTGN ROLAND VGRNN TGCN GCLSTM EvolveGCN TopoGED
noitaulavEegdE
oo-bankPrecision↑ 0.00±0.00 0.01±0.10 OOM 0.00±0.00 0.00±0.03 0.00±0.00 0.14±0.09
oo-bankRecall↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.17±0.11
oo-bankF1↑ 0.00±0.00 0.01±0.01 OOM 0.00±0.00 0.01±0.01 0.00±0.00 0.15±0.08
oo-nobankPrecision↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.03
oo-nobankRecall↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.02
oo-nobankF1↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.02
EdgePrecision↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.08±0.05
EdgeRecall↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.08±0.05
EdgeF1↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.08±0.05
Numo-nPredicted→0 −0.71±0.13 −0.65±0.43 OOM −0.61±0.37 0.01±1.20 −0.63±0.23 0.17±0.49
Numn-nPredicted→0 186.38±164.41 188.90±166.41 OOM 184.65±168.51 217.60±170.96 183.59±164.55 0.49±1.55
noitaulavEedoN PrecisionNodes↑ 0.00±0.00 0.02±0.02 OOM 0.00±0.00 0.02±0.01 0.00±0.00 0.53±0.11
RecallNodes↑ 0.00±0.00 0.01±0.01 OOM 0.00±0.00 0.02±0.01 0.01±0.01 0.56±0.13
F1Nodes↑ 0.00±0.00 0.01±0.01 OOM 0.00±0.00 0.02±0.01 0.01±0.01 0.53±0.09
PrecisionOldNodes↑ 0.02±0.05 0.42±0.38 OOM 0.00±0.00 0.37±0.19 0.15±0.13 0.33±0.09
RecallOldNodes↑ 0.00±0.01 0.01±0.01 OOM 0.00±0.00 0.04±0.02 0.02±0.02 0.34±0.09
F1OldNodes↑ 0.00±0.01 0.03±0.03 OOM 0.00±0.00 0.07±0.04 0.04±0.03 0.32±0.07
NewNodesPredicted↑ 8.39±3.56 0.33±0.59 OOM 0.42±0.54 1.67±1.20 6.73±3.08 0.14±0.44
noitaulavEerutcurtS
AvgNodeDegree→0 −0.27±0.09 4.46±1.86 OOM 3.89±1.18 2.38±0.94 −0.12±0.11 0.02±0.10
UniqueDegreeCount→0 −0.23±0.19 1.57±0.93 OOM 1.79±0.75 1.47±0.67 −0.02±0.35 0.05±0.32
DegreeCentrality→0 −0.78±0.09 11.56±7.42 OOM 9.02±5.27 2.77±2.39 −0.67±0.15 0.06±0.56
AssortativityCoefficient→0 −2.69±0.89 1.48±1.62 OOM 1.42±1.50 1.39±1.95 −2.46±1.33 1.04±0.99
ClusteringCoefficient→0 5.10±16.92 48.00±88.92 OOM 51.54±123.07 48.06±121.80 12.61±34.74 3.06±7.58
Density→0 −0.78±0.09 11.56±7.42 OOM 9.02±5.27 2.77±2.39 −0.67±0.15 0.06±0.56
NumTriangles→0 28.25±79.49 989.03±1802.47 OOM 953.29±1918.38 797.07±1384.43 59.66±82.10 7.17±9.37
DescriptorNorm→0 1033.25±583.15 364.97±280.28 OOM 350.06±283.24 360.46±264.26 761.41±597.24 180.59±205.62
MedianExtraNodes→0 297.00±19.20 0.00±0.00 OOM 0.00±0.00 7.00±3.74 222.00±18.75 12.00±3.60
MedianMissingNodes→0 0.00±0.00 52.00±3.64 OOM 50.00±5.57 0.00±1.00 0.00±0.00 0.00±0.06
MedianExtraEdges→0 157.00±13.94 148.00±15.52 OOM 154.00±13.97 232.00±14.32 155.00±14.01 13.00±3.87
MedianMissingEdges→0 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.13
Table22: Comprehensiveevaluationresults—includingedge,node,andstructuralmetrics—forall
benchmarkmodelsonIconomi. →0indicatesthatvaluescloserto0arebetter. ↑denotesthathigher
valuesarebetter,while↓denotesthatlowervaluesarebetter. Boldindicatesbest,thesecond-bestis
underlined. OOM indicateout-of-memory.
Metric HTGN ROLAND VGRNN TGCN GCLSTM EvolveGCN TopoGED
noitaulavEegdE
oo-bankPrecision↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.19±0.10
oo-bankRecall↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.28±0.15
oo-bankF1↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.22±0.09
oo-nobankPrecision↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.03
oo-nobankRecall↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.02
oo-nobankF1↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.01±0.02
EdgePrecision↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.10±0.06
EdgeRecall↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.11±0.06
EdgeF1↑ 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.10±0.06
Numo-nPredicted→0 −0.58±0.41 −0.79±0.32 OOM 0.02±0.63 −0.53±0.54 −0.75±0.36 0.18±0.54
Numn-nPredicted→0 157.28±145.81 160.25±149.02 OOM 133.04±134.74 303.04±233.97 175.42±285.77 0.38±1.37
noitaulavEedoN PrecisionNodes↑ 0.00±0.00 0.01±0.01 OOM 0.00±0.00 0.01±0.00 0.00±0.00 0.52±0.11
RecallNodes↑ 0.00±0.00 0.01±0.01 OOM 0.00±0.00 0.03±0.01 0.01±0.01 0.54±0.11
F1Nodes↑ 0.00±0.00 0.01±0.01 OOM 0.00±0.00 0.01±0.00 0.00±0.01 0.51±0.09
PrecisionOldNodes↑ 0.00±0.02 0.57±0.45 OOM 0.00±0.00 0.26±0.07 0.09±0.16 0.28±0.07
RecallOldNodes↑ 0.00±0.00 0.02±0.01 OOM 0.00±0.00 0.06±0.02 0.01±0.02 0.30±0.08
F1OldNodes↑ 0.00±0.01 0.03±0.02 OOM 0.00±0.00 0.09±0.03 0.02±0.04 0.28±0.06
NewNodesPredicted↑ 10.24±7.16 0.79±1.01 OOM 0.49±0.71 7.82±3.90 6.63±6.09 0.15±0.50
noitaulavEerutcurtS
AvgNodeDegree→0 −0.37±0.06 2.82±0.72 OOM 3.41±0.95 0.59±1.16 −0.04±0.15 0.02±0.09
UniqueDegreeCount→0 −0.53±0.13 1.65±0.84 OOM 1.89±0.82 −0.24±0.94 0.07±0.35 0.10±0.28
DegreeCentrality→0 −0.83±0.14 5.28±4.95 OOM 7.30±5.43 −0.48±0.56 −0.61±0.26 0.07±0.81
AssortativityCoefficient→0 −2.26±0.49 0.06±0.37 OOM 0.92±0.55 1.59±0.93 −1.66±0.81 0.61±0.52
ClusteringCoefficient→0 1.36±3.48 19.54±34.79 OOM 25.50±47.62 10.55±29.54 7.60±17.10 2.74±7.26
Density→0 −0.83±0.14 5.28±4.95 OOM 7.30±5.43 −0.48±0.56 −0.61±0.26 0.07±0.81
NumTriangles→0 5.91±18.95 448.38±863.70 OOM 488.80±1095.20 281.67±1744.48 44.18±110.32 4.81±3.54
DescriptorNorm→0 935.66±592.46 222.01±227.53 OOM 217.45±219.85 658.49±202.90 452.03±566.43 122.31±195.08
MedianExtraNodes→0 237.00±15.06 0.00±0.00 OOM 0.00±0.00 197.00±7.56 129.00±10.25 5.00±2.85
MedianMissingNodes→0 0.00±0.00 23.00±3.02 OOM 32.00±2.32 0.00±0.00 0.00±0.00 0.00±0.54
MedianExtraEdges→0 110.00±8.97 96.00±6.98 OOM 106.00±8.67 267.00±15.10 109.00±8.44 5.00±2.72
MedianMissingEdges→0 0.00±0.00 0.00±0.00 OOM 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.38
30

Table23: Comprehensiveevaluationresults—includingedge,node,andstructuralmetrics—forall
benchmarkmodelsonRedditB.→0indicatesthatvaluescloserto0arebetter. ↑denotesthathigher
valuesarebetter,while↓denotesthatlowervaluesarebetter. Boldindicatesbest,thesecond-bestis
underlined.
Metric HTGN ROLAND VGRNN TGCN GCLSTM EvolveGCN TopoGED
noitaulavEegdE
oo-bankPrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.05±0.03
oo-bankRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.10±0.08
oo-bankF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.06±0.03
oo-nobankPrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.01
oo-nobankRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.01
oo-nobankF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.01
EdgePrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.03±0.02
EdgeRecall↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.03±0.02
EdgeF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.03±0.02
Numo-nPredicted→0 −1.00±0.00 −1.00±0.00 −1.00±0.00 −1.00±0.00 −1.00±0.00 −1.00±0.00 0.11±0.51
Numn-nPredicted→0 467.53±284.41 461.78±269.78 866.86±528.87 466.79±283.91 782.58±470.41 466.82±283.93 0.59±1.56
noitaulavEedoN PrecisionNodes↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.38±0.04
RecallNodes↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.38±0.03
F1Nodes↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.38±0.03
PrecisionOldNodes↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.34±0.04
RecallOldNodes↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.34±0.03
F1OldNodes↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.34±0.03
NewNodesPredicted↑ 23.99±6.86 4.88±2.47 3.82±1.78 6.87±5.81 3.22±1.11 8.53±4.13 0.04±0.31
noitaulavEerutcurtS
AvgNodeDegree→0 0.41±0.16 5.55±2.10 13.17±2.96 5.09±2.65 13.02±1.71 3.01±1.05 0.02±0.08
UniqueDegreeCount→0 1.65±0.55 2.71±0.86 2.13±0.81 2.31±1.14 4.63±0.98 2.78±0.90 0.87±0.39
DegreeCentrality→0 −0.22±0.15 17.19±9.94 43.45±16.60 16.15±10.87 45.20±7.95 5.70±3.12 0.03±0.11
AssortativityCoefficient→0 −3.63±28.88 6.25±65.25 10.71±77.63 8.28±67.75 5.16±60.15 1.00±21.05 1.10±32.12
ClusteringCoefficient→0 3.78±13.51 44.47±110.98 58.99±156.80 42.99±161.65 46.75±133.31 18.76±52.63 2.07±6.25
Density→0 −0.22±0.15 17.19±9.94 43.45±16.60 16.15±10.87 45.20±7.95 5.70±3.12 0.03±0.11
NumTriangles→0 151.71±211.40 843.41±1136.71 1707.68±2023.11 687.13±934.66 1970.28±2612.84 669.92±789.01 23.25±38.07
DescriptorNorm→0 1046.93±276.60 1002.78±148.13 1495.89±206.22 1030.84±319.63 1421.72±174.75 755.22±126.19 446.58±67.84
MedianExtraNodes→0 352.00±16.18 4.00±0.00 0.00±0.00 269.00±0.00 0.00±0.00 71.00±0.00 23.00±4.24
MedianMissingNodes→0 0.00±0.00 253.00±10.95 291.00±6.25 265.00±9.92 299.00±8.01 161.00±19.08 0.00±2.24
MedianExtraEdges→0 463.00±20.23 460.00±20.39 1101.00±21.94 462.00±21.05 967.00±17.34 462.00±20.31 13.00±7.77
MedianMissingEdges→0 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±1.38
Table24: Comprehensiveevaluationresults—includingedge,node,andstructuralmetrics—forall
benchmark models on TGBL-Wiki. → 0 indicates that values closer to 0 are better. ↑ denotes
thathighervaluesarebetter,while↓denotesthatlowervaluesarebetter. Boldindicatesbest,the
second-bestisunderlined.
Metric HTGN ROLAND VGRNN TGCN GCLSTM EvolveGCN TopoGED
noitaulavEegdE
oo-bankPrecision↑ 0.39±0.23 0.24±0.15 0.00±0.00 0.65±0.40 0.00±0.00 0.00±0.00 0.27±0.16
oo-bankRecall↑ 0.03±0.02 0.01±0.01 0.00±0.00 0.01±0.01 0.00±0.00 0.00±0.00 0.34±0.20
oo-bankF1↑ 0.05±0.03 0.02±0.01 0.00±0.00 0.02±0.01 0.00±0.00 0.00±0.00 0.30±0.18
oo-nobankPrecision↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.07±0.07
oo-nobankRecall↑ 0.01±0.01 0.01±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.07±0.08
oo-nobankF1↑ 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.07±0.08
EdgePrecision↑ 0.01±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.17±0.01
EdgeRecall↑ 0.01±0.00 0.01±0.00 0.00±0.00 0.01±0.00 0.00±0.00 0.00±0.00 0.18±0.01
EdgeF1↑ 0.01±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.18±0.01
Numo-nPredicted→0 −0.89±0.03 −1.00±0.00 −1.00±0.00 −0.94±0.10 0.90±0.24 −1.00±0.00 0.14±0.18
Numn-nPredicted→0 0.05±0.94 −0.60±0.55 −0.60±0.55 −0.60±0.55 14.76±12.79 −0.60±0.55 3.09±5.06
noitaulavEedoN PrecisionNodes↑ 0.57±0.03 0.33±0.05 0.11±0.01 0.34±0.03 0.53±0.09 0.20±0.03 0.63±0.03
RecallNodes↑ 0.25±0.01 0.18±0.01 0.48±0.03 0.13±0.08 0.08±0.01 0.18±0.08 0.67±0.02
F1Nodes↑ 0.35±0.01 0.24±0.02 0.17±0.01 0.17±0.07 0.13±0.02 0.17±0.05 0.65±0.01
PrecisionOldNodes↑ 0.57±0.03 0.33±0.05 0.11±0.01 0.34±0.02 0.57±0.06 0.20±0.03 0.60±0.02
RecallOldNodes↑ 0.28±0.01 0.21±0.02 0.55±0.04 0.15±0.09 0.08±0.01 0.20±0.10 0.63±0.02
F1OldNodes↑ 0.38±0.01 0.25±0.02 0.18±0.01 0.19±0.08 0.14±0.01 0.19±0.05 0.62±0.01
NewNodesPredicted↑ −0.98±0.00 −1.00±0.00 −1.00±0.00 −0.99±0.01 −0.86±0.02 −1.00±0.00 0.14±0.18
noitaulavEerutcurtS
AvgNodeDegree→0 5.13±0.40 4.02±0.63 0.14±0.02 8.12±4.85 17.51±1.16 2.59±2.12 0.01±0.02
UniqueDegreeCount→0 1.42±0.20 2.24±0.32 −0.90±0.01 2.35±0.40 2.92±0.23 1.51±0.55 0.26±0.19
DegreeCentrality→0 13.11±1.64 8.38±2.43 −0.75±0.02 38.96±33.94 130.76±13.01 5.27±7.07 −0.05±0.06
AssortativityCoefficient→0 −6.90±1.44 3.51±1.32 8.97±2.27 3.91±0.48 4.27±1.21 3.80±1.48 1.58±0.35
ClusteringCoefficient→0 0.36±0.01 0.00±0.00 0.00±0.00 0.56±0.20 0.79±0.01 0.41±0.28 0.02±0.01
Density→0 13.11±1.64 8.38±2.43 −0.75±0.02 38.96±33.94 130.76±13.01 5.27±7.07 −0.05±0.06
NumTriangles→0 11045.00±1653.77 0.00±0.00 0.00±0.00 30045.00±13822.67 40188.20±6563.66 11052.40±4986.92 99.60±39.51
DescriptorNorm→0 3447.24±307.45 2997.41±242.59 17861.99±1923.61 3205.85±759.27 3762.26±296.79 2652.77±461.59 1034.67±135.03
MedianExtraNodes→0 0.00±0.00 0.00±0.00 5104.00±414.76 0.00±0.00 0.00±0.00 0.00±203.32 99.00±51.77
MedianMissingNodes→0 753.00±72.77 646.00±77.90 0.00±0.00 1008.00±294.69 1154.00±66.14 77.00±257.93 0.00±4.34
MedianExtraEdges→0 2123.00±176.71 2257.00±117.44 5292.00±418.19 2070.00±185.88 2070.00±188.06 2070.00±185.93 87.00±60.57
MedianMissingEdges→0 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±0.00 0.00±1.68
G AdditionalResults
869
Figure7visualizeswhyfixededgepriorsfailontemporaldatasets. Theempiricalmixofo-obank,
870
o-onobank,o-n,andn-nedgesshiftsovertime,soamodelthattreatsedgeformationasstationary
871
willallocateprobabilitymasstothewrongedgetypesduringregimechanges.
872
31

1.0
0.8
0.6
0.4
0.2
0.0
0 50 100 150 200 250 300 350 400
Snapshot
ytilibaborP
laciripmE
o-obank
n-n
o-n
o-onobank
Figure7: EmpiricalprobabilitiesofedgetypesovertimefortheReddit-Bnetwork.
32

NeurIPSPaperChecklist
873
1. Claims
874
Question: Dothemainclaimsmadeintheabstractandintroductionaccuratelyreflectthe
875
paper’scontributionsandscope?
876
Answer: [Yes]
877
Justification: The abstract and introduction consistently present the work as an inverse-
878
topologyapproachtotemporalgraphforecastingthatmodelsnodechurnandglobalstructure.
879
Theseclaimsarewellalignedwiththepaper’scontributions,scope,andempiricalevaluation.
880
2. Limitations
881
Question: Doesthepaperdiscussthelimitationsoftheworkperformedbytheauthors?
882
Answer: [Yes]
883
Justification: WediscussthelimitationofourworkattheendofSection5.
884
3. Theoryassumptionsandproofs
885
Question: Foreachtheoreticalresult,doesthepaperprovidethefullsetofassumptionsand
886
acomplete(andcorrect)proof?
887
Answer: [Yes]
888
Justification: WeprovideProposition4.1 tosupportour claim thattopologydescriptors
889
usedinourmethodarestableforecastingtargets. Weprovidethecorrespondingproofin
890
SectionE.
891
4. Experimentalresultreproducibility
892
Question: Doesthepaperfullydisclosealltheinformationneededtoreproducethemainex-
893
perimentalresultsofthepapertotheextentthatitaffectsthemainclaimsand/orconclusions
894
ofthepaper(regardlessofwhetherthecodeanddataareprovidedornot)?
895
Answer: [Yes]
896
Justification: Thepaperprovidesacomprehensivedescriptionofthedatasets,modelconfig-
897
urations,andhyperparametersused,ensuringthattheexperimentalresultsarereproducible.
898
A link to an Anonymous Repository containing code for reproducing is provided under
899
Section5.
900
5. Openaccesstodataandcode
901
Question: Doesthepaperprovideopenaccesstothedataandcode,withsufficientinstruc-
902
tionstofaithfullyreproducethemainexperimentalresults,asdescribedinsupplemental
903
material?
904
Answer: [Yes]
905
Justification: Code to reproduce results is publicly available at https://anonymous.
906
4open.science/r/TopoGED/, with detailed instructions provided in the supplemental
907
materialtoensurereproducibility.
908
6. Experimentalsetting/details
909
Question: Doesthepaperspecifyallthetrainingandtestdetails(e.g.,datasplits,hyperpa-
910
rameters,howtheywerechosen,typeofoptimizer)necessarytounderstandtheresults?
911
Answer: [Yes]
912
Justification: Choicesofhyperparameters(includingoptimizer,scheduler,etc.) foreach
913
model used in our paper can be found in Section 5. For each dataset, data splits can
914
beretrievedfromrunningthecodeattheAnonymousRepository(https://anonymous.
915
4open.science/r/GAB/README.md)
916
7. Experimentstatisticalsignificance
917
Question:Doesthepaperreporterrorbarssuitablyandcorrectlydefinedorotherappropriate
918
informationaboutthestatisticalsignificanceoftheexperiments?
919
Answer: [Yes]
920
33

Justification: Weshowresultsofthreerunsalongwithstandarddeviations,providingclear
921
informationonthestatisticalsignificanceoftheexperiments.
922
8. Experimentscomputeresources
923
Question: Foreachexperiment,doesthepaperprovidesufficientinformationonthecom-
924
puterresources(typeofcomputeworkers,memory,timeofexecution)neededtoreproduce
925
theexperiments?
926
Answer: [Yes]
927
Justification: Thepaperdetailsthecomputeresourcesusedfortheexperiments,including
928
thetypeofhardware,memory,andtimerequiredforexecution,ensuringtransparencyand
929
reproducibility,Section5.
930
9. Codeofethics
931
Question: Doestheresearchconductedinthepaperconform, ineveryrespect, withthe
932
NeurIPSCodeofEthicshttps://neurips.cc/public/EthicsGuidelines?
933
Answer: [Yes]
934
Justification: TheauthorshavereadandconfirmedthattheresearchadherestotheNeurIPS
935
CodeofEthics,withconsiderationsforresponsibleAIpracticesandtransparencyinthe
936
methodologiesanddataused.
937
10. Broaderimpacts
938
Question: Does the paper discuss both potential positive societal impacts and negative
939
societalimpactsoftheworkperformed?
940
Answer: [Yes]
941
Justification: BroaderImpactisdiscussedinSecionA
942
11. Safeguards
943
Question: Doesthepaperdescribesafeguardsthathavebeenputinplaceforresponsible
944
releaseofdataormodelsthathaveahighriskformisuse(e.g.,pre-trainedlanguagemodels,
945
imagegenerators,orscrapeddatasets)?
946
Answer: [N/A]
947
Justification: Thepaperdoesnotinvolvethereleaseofmodelsordatathathaveahighrisk
948
ofmisuse,andthusthisquestionisnotapplicable.
949
12. Licensesforexistingassets
950
Question: Arethecreatorsororiginalownersofassets(e.g.,code,data,models),usedin
951
thepaper,properlycreditedandarethelicenseandtermsofuseexplicitlymentionedand
952
properlyrespected?
953
Answer: [Yes]
954
Justification: Thepapercreditsthecreatorsofallassetsused,includingdatasetsandcode,
955
andadherestotherespectivelicensesandtermsofuse.
956
13. Newassets
957
Question:Arenewassetsintroducedinthepaperwelldocumentedandisthedocumentation
958
providedalongsidetheassets?
959
Answer: [N/A]
960
Justification: Wedonotintroduceanynewassetsinthispaper.
961
14. Crowdsourcingandresearchwithhumansubjects
962
Question: Forcrowdsourcingexperimentsandresearchwithhumansubjects,doesthepaper
963
includethefulltextofinstructionsgiventoparticipantsandscreenshots,ifapplicable,as
964
wellasdetailsaboutcompensation(ifany)?
965
Answer: [N/A]
966
Justification: Thepaperdoesnotinvolvecrowdsourcingorresearchwithhumansubjects.
967
15. Institutional review board (IRB) approvals or equivalent for research with human
968
subjects
969
34

Question: Doesthepaperdescribepotentialrisksincurredbystudyparticipants,whether
970
suchrisksweredisclosedtothesubjects,andwhetherInstitutionalReviewBoard(IRB)
971
approvals(oranequivalentapproval/reviewbasedontherequirementsofyourcountryor
972
institution)wereobtained?
973
Answer: [N/A]
974
Justification: Theresearchdoesnotinvolvehumansubjects,andtherefore,IRBapprovalis
975
notapplicable.
976
16. DeclarationofLLMusage
977
Question: Does the paper describe the usage of LLMs if it is an important, original, or
978
non-standardcomponentofthecoremethodsinthisresearch? NotethatiftheLLMisused
979
onlyforwriting,editing,orformattingpurposesanddoesnotimpactthecoremethodology,
980
scientificrigor,ororiginalityoftheresearch,declarationisnotrequired.
981
Answer: [N/A]
982
Justification: Theresearchdoesn’tinvolveLLMs
983
35
