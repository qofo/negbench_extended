### 종합 평가 및 판정 (Overall Recommendation)

**판정: Reject (거절)**

본 논문은 CLIP 이중 인코더의 부정(Negation) 매칭 실패 원인을 $2 \times 2$ 요인 분해(Factor Decomposition)를 통해 대수적으로 규명하고, 실패의 본질이 표현 부재가 아닌 지배적 주효과($\alpha, \beta$) 대비 극도로 작은 교차항($\gamma$)에 있음을 밝힙니다. 이론적 프레임워크와 진단적 착안은 매우 우수하나, **극히 제한된 인공적 실험 환경(AB-Swap), 실제 $1:N$ 검색으로의 확장성 검증 부족, 제안된 대안(Rank-1 Bilinear)의 낮은 절대 성능** 등 핵심적인 약점으로 인해 현재 형태로는 게재를 승인하기 어렵습니다.

---

### 논문 요약 (Summary)

본 논문은 CLIP의 부정 질의 매칭 실패를 분석하기 위해 $2 \times 2$ 최소쌍(Minimal Pair) 유사도를 평균 $C$, 이미지 주효과 $\beta$, 텍스트 주효과 $\alpha$, 교차항 $\gamma$로 분해하는 좌표 변환 프레임워크를 제안합니다:


$$S_{ab} = C + a\beta + b\alpha + ab\gamma \quad (a, b \in \{+1, -1\})$$


저자들은 부정 검색/선택의 성공(Winoground의 Group Score와 동치)이 $\gamma > \max(\vert{}\alpha\vert{}, \vert{}\beta\vert{})$와 동치임을 증명합니다. ViT-B/32 모델 실험 결과, 프로브 정확도(AUC 0.759~0.945)로 확인되는 정보가 인코더에 존재함에도 불구하고, 교차항 $\gamma$의 크기가 주효과의 1/5.2 수준에 불과하여 2×2 정답률이 4.03%(우연 확률 16.67%)로 붕괴함을 실증했습니다.

---

### 주요 강점 (Strengths)

* **명확한 대수적 환원:** Winoground류의 $2 \times 2$ 매칭 성공 조건을 가정 없는 요인 분해를 통해 $\gamma > \max(\vert{}\alpha\vert{}, \vert{}\beta\vert{})$라는 부등식으로 명료하게 정립했습니다.


* **통제된 최소쌍 설계:** BEAF 기반 인페인팅 이미지와 어휘 다중집합을 완전히 일치시킨 AB-swap 텍스트를 구성하여 BoW(Bag-of-Words) 편향을 효과적으로 차단하고 순수 결합 정보를 측정했습니다.


* **표현과 유사도 간 괴리 입증:** 프로브 법선 외적 기반의 Rank-1 Bilinear 헤드($v^T(w_I w_T^T)t$)만으로 코사인 대비 성능이 4.03%에서 24.96%로 상승함을 보여, 유사도 계산 구조의 병목을 실증적으로 짚어냈습니다.



---

### 주요 약점 및 거절 사유 (Weaknesses & Rejection Reasons)

**1. 지나치게 인공적인 AB-Swap 평가 환경과 실용적 일반화 한계**

* 본 연구가 사용하는 AB-swap 텍스트 ("A features a pizza, but lacks a cup" vs. "A features a cup, but lacks a pizza")는 어휘를 통제하기 위한 극단적 템플릿입니다.


* 그러나 실제 검색 환경이나 표준 부정 벤치마크(NegBench, CC-Neg, VALSE)의 질의는 비대칭적 단일 객체 부재("a photo of a room with no chairs", "dog not on grass")가 대다수입니다.


* 저자 스스로 한계에서 "단일 객체 부정을 다루지 못한다"고 인정했듯이, 상쇄할 대조 객체가 없는 일반 질의에서 본 분해 이론이 어떻게 확장될 수 있는지 제시하지 못해 분석의 유효 범위가 크게 축소됩니다.



**2. $2 \times 2$ 블록 평가에서 $1:N$ 실제 검색으로의 비약**

* 본 논문의 핵심 정리인 $\gamma > \max(\vert{}\alpha\vert{}, \vert{}\beta\vert{})$는 $2 \times 2$ 블록 내 평가(Winoground Group Score)에 국한됩니다.


* 실제 대규모 갤러리 검색($1:N$)에서는 $2 \times 2$ 짝이 주어지지 않으며, 갤러리 전체의 분산과 배경 편향이 개입합니다.


* 4절에서 COCO 5,000장 R@1 하락폭과의 상관관계($r=+0.835$)를 언급했으나, 2페이지 지면 한계로 인해 $2 \times 2$ 분석 틀이 $1:N$ 순위화(Ranking) 메커니즘으로 어떻게 대수적으로 연결되는지 충분한 수학적/실험적 근거가 제시되지 않았습니다.



**3. 제안된 대안(Rank-1 Bilinear)의 낮은 절대 성능 및 기존 연구 대비 실효성 부족**

* 저자들은 코사인 유사도(4.03%)의 한계를 극복하는 대안으로 Rank-1 Bilinear 헤드(24.96%)와 2비트 합성(26.13%)을 제시합니다.


* 그러나 4지선다 우연 확률(16.67%)을 감안할 때 24~26%는 여전히 실용적으로 매우 저조한 수치입니다.


* 최근 연구들(예: SpaceVLM, Vector Arithmetic, PeakPatch, NegToMe)은 별도의 학습 없이도 기하학적 부분공간(Subspace) 투영이나 테스트 타임 수정을 통해 NegBench MCQ 등에서 60~70% 이상의 정확도를 달성하고 있습니다. 이러한 최신 훈련 불필요(Training-free) 방법론들과의 비교 및 제안 메커니즘의 실질적 우위가 결여되어 있습니다.



**4. 축약본(2-page)에 따른 검증 데이터 부족**

* 본문 서두에 "아홉 모델 비교, 개입 실험, 외부 벤치마크는 완결 논문(PAPER.md)에 있다"고 명시하고 있으나, 심사는 제출된 2페이지 문서 자체로 평가되어야 합니다.


* 본문에는 단일 모델(ViT-B/32)에 대한 수치만 제시되어 있어, SigLIP, NegCLIP, ConCLIP 등 다양한 목적함수와 아키텍처에서 이 분해 규칙이 보편적으로 성립하는지 본 논문 내에서 독립적으로 검증할 수 없습니다.



---

### 질문 및 수정 요구사항 (Questions for Authors)

* **질문 1 (비대칭 단일 부정으로의 확장):** "a photo with no cat"처럼 대칭되는 객체 쌍이 없는 일반적인 부정 질의에 대해 본 요인 분해($S_{ab} = C + a\beta + b\alpha + ab\gamma$)를 어떻게 정의하고 계측할 수 있습니까?


* **질문 2 (최신 기하학적 방법론과의 관계):** SpaceVLM이나 Aggarwal et al.의 벡터 연산 기반 수정이 본 논문의 분해 관점에서는 $\gamma$를 증폭시키는 것인지, 아니면 $\alpha, \beta$ 주효과를 억제하는 것인지 대수적으로 설명할 수 있습니까?


* **질문 3 (수식 표기 오류):** 2절 및 3절 본문 내 일부 수식 기호(예: 2절 'Sab Caẞ bo aby', 3.2절 '$Y=0.001210/5$', '378개중 368개(97.4%)이고, 성공 조건을 만족하는 조합은 378개 중 0개다' 문맥의 기호 누락)의 렌더링이 깨져 있습니다. 정확한 수식 표기를 확인 바랍니다.


리뷰어 관점에서 보면, 이 논문은 “아이디어가 없는 논문”은 아닙니다. 오히려 핵심 관찰은 꽤 흥미롭습니다. 다만 현재 2페이지 판본 그대로라면 저는 **Weak Reject에 가깝게 평가**하겠습니다. 이유는 실험 결과가 약해서가 아니라, 가장 중요한 novelty claim이 아직 수학적 재표현과 실증적 발견 사이에서 충분히 분리되지 않았기 때문입니다.

논문의 핵심 주장은 다음과 같이 이해했습니다. 기존 연구들은 “negation 정보가 encoder에 존재하고, 이후 representation alignment/interface가 문제”라는 방향으로 접근해 왔는데, 이 논문은 동일 장면의 객체 제거 최소쌍을 사용하여 이미지 측 객체 정보가 실제로 존재하더라도 그 신호의 크기가 충분하지 않을 수 있음을 보이고, 2×2 similarity를 주효과 α, β와 interaction γ로 분해하여 negation matching의 성공 조건을 정량화한다는 것입니다. 실제 측정에서는 |α|=0.00781, |β|=0.00464, γ=0.00056으로, γ가 주효과보다 훨씬 작습니다. 

## 총평

**판정: Weak Reject / Borderline**

**장점**

* 문제를 단순히 “CLIP이 negation을 못 한다”에서 “어떤 신호가 얼마나 강한가”로 바꾸려는 시도는 좋습니다.
* 동일 장면에서 객체만 제거하는 최소쌍은 기존의 서로 다른 자연 이미지 간 probing보다 인과적 해석에 훨씬 유리합니다.
* α, β, γ의 분해는 명확하고 재현하기 쉽습니다.
* 실제로 γ가 존재하지만 매우 작다는 결과는 흥미롭습니다.
* 특히 γ > max(|α|,|β|)라는 조건과 실제 2×2 결과가 100% 일치한다는 점은 분석적으로 깔끔합니다. 

**하지만**

* 핵심 수학적 정리는 사실상 2×2 행렬의 Hadamard/Fourier decomposition에 대한 자명한 좌표변환입니다.
* 따라서 “새로운 이론”이라고 주장하기 어렵습니다.
* 논문의 진짜 novelty는 수학이 아니라 **객체 단위 통제 하에서 γ의 상대적 크기를 실제로 측정했다는 실증적 발견**이어야 합니다.
* 그런데 현재 서술은 수학적 명제가 너무 전면에 나와 있어 오히려 논문의 novelty를 약화시킵니다.
* 더 심각하게는 기존 intervention에 대한 “도달 불가능성” 주장이 현재 증거보다 강합니다.
* 특히 “정렬해도 최대 4.6배”, “따라서 기존 계열은 원리적으로 불가능” 같은 문장은 리뷰어가 공격하기 매우 쉽습니다.

---

# 1. 가장 큰 문제: Proposition 1은 novelty가 거의 없다

논문의 중심 명제는

$$
\Delta(S)=2\gamma-2\max(|\alpha|,|\beta|)
$$

이고 따라서

$$
\Delta(S)>0
\iff
\gamma>\max(|\alpha|,|\beta|)
$$

입니다. 

수학적으로는 맞습니다.

하지만 리뷰어 입장에서는 바로 이렇게 질문합니다.

> “이것이 정말 연구 기여인가?”

2×2 matrix에 네 개의 값이 있고,

$$
S_{ab}=C+a\beta+b\alpha+ab\gamma
$$

로 표현하는 것은 네 개의 자유도를 네 개의 basis coefficient로 바꾸는 것뿐입니다. 논문 스스로도 이것을 “2×2 Hadamard transform”이라고 설명하고 있습니다. 

즉,

> “네 개의 similarity 값을 main effect와 interaction effect로 재parameterize했다.”

는 것에 가깝습니다.

이 자체는 논문을 지탱하기에는 약합니다.

### 리뷰어가 할 공격

“Interaction term이 중요하다는 것은 2×2 factorial design의 표준적인 사실이다. 저자들은 기존 CLIP의 negation problem에 이를 적용했을 뿐이며, 새로운 이론적 결과라고 보기 어렵다.”

이 공격에는 상당히 취약합니다.

따라서 Proposition 1을 **주요 novelty로 내세우면 안 됩니다.**

오히려 다음처럼 위치시켜야 합니다.

> “We introduce a controlled 2×2 diagnostic that decomposes negation matching into image main effect, text main effect, and cross-modal interaction, allowing us to empirically quantify which component limits retrieval.”

즉 **수학은 도구이고, 발견은 empirical finding**이어야 합니다.

---

# 2. 진짜 좋은 부분은 γ 자체가 아니라 “γ가 너무 작다”는 결과다

논문에서 가장 흥미로운 숫자는 사실 이겁니다.

$$
|\alpha|=0.00781,\quad
|\beta|=0.00464,\quad
\gamma=0.00056.
$$

즉

$$
|\alpha|/\gamma\approx14.1.
$$

논문도 이 점을 강조합니다. 33개 객체 모두에서 \(|\alpha|>\gamma\), 그리고 어느 객체에서도 \(\gamma>|\beta|\)가 나타나지 않았습니다. 

이건 상당히 좋은 empirical observation입니다.

왜냐하면 기존 연구의 질문은 대체로

> “negation information이 representation에 존재하는가?”

였기 때문입니다.

그런데 이 논문은 질문을

> “존재한다면, 실제 matching decision을 바꿀 정도로 강한가?”

로 바꿉니다.

이 framing은 기존 probing 결과와 잘 대비됩니다.

예를 들어 기존 연구는 CLIP 내부에서 negation 관련 representation이나 direction을 찾거나, embedding을 조작하면 성능이 좋아질 수 있음을 보였습니다. Quantmeyer et al.은 CLIP text encoder 내부에서 negation 처리를 분석했고,  Sammani et al.은 embedding space에 negation direction이 존재한다고 보고했습니다. 

따라서 논문의 가장 좋은 메시지는

> **“Representable does not imply decision-relevant.”**

입니다.

이 한 문장이 논문 전체를 지탱해야 합니다.

---

# 3. 그러나 “기존 연구들이 모두 같은 전제를 가진다”는 표현은 너무 강하다

논문은 기존 연구들을 대략 다음과 같이 묶습니다.

* information은 encoder에 존재한다.
* 실패는 representation/alignment/interface에서 발생한다.

그리고 본 연구는 그 전제를 문제 삼습니다.

그런데 이건 상당히 조심해야 합니다.

실제로 업로드된 선행연구들을 보면 서로 원인에 대한 주장이 상당히 다릅니다.

예를 들어:

* Quantmeyer et al.은 CLIP 내부의 negation 처리 위치와 attention head를 분석합니다. 
* NegBench는 negation data 부족과 contrastive training의 문제를 강조합니다. 
* NegationCLIP은 negation-rich data로 fine-tuning합니다. 
* SpaceVLM은 negation을 point가 아니라 subspace로 모델링해야 한다고 주장합니다. 
* ICLR 2026의 “Seeing What’s Not There”는 text embedding에서 negated semantic component를 직접 제거합니다. 
* “When Negation Is a Geometry Problem”은 아예 CLIP embedding geometry와 negation direction을 분석합니다. 
* PeakPatch는 intermediate feature에 negation signal이 존재하지만 final representation에서 collapse된다고 주장합니다. 

따라서

> “기존 연구들은 모두 information이 충분히 표현되어 있다고 가정한다.”

는 문장은 리뷰어가 쉽게 반박할 수 있습니다.

더 정확한 표현은

> “Prior probing and intervention results demonstrate that negation-related information can be recovered or manipulated in CLIP representations. However, these results do not establish that the information is sufficiently strong at the final cross-modal similarity interface to determine retrieval rankings.”

정도입니다.

이렇게 하면 기존 연구를 부정하지 않고 **그 연구들이 대답하지 않은 질문을 제기하는 형태**가 됩니다.

---

# 4. 가장 위험한 부분: 0.88%와 16.7% 비교

논문은 실제 2×2 matching accuracy가

> 12 / 1357 = 0.88%

이고 random이

> 16.7%

라고 합니다. 

이 결과는 눈에 띕니다.

하지만 “CLIP이 random보다 훨씬 나쁘다”라고 강조하면 위험합니다.

왜냐하면 이 2×2 task는 일반적인 retrieval benchmark가 아니라 **저자들이 구성한 매우 특수한 최소쌍 task**이기 때문입니다.

더구나 4개의 similarity가 독립적인 random variable이라는 가정에서 16.7%를 계산한 것입니다. 

실제 CLIP similarity는 독립 random variable이 아닙니다.

따라서 16.7%는

> “chance under arbitrary random ordering”

이지,

> “CLIP이 random guess보다 못하다”

라는 강한 의미의 baseline이 아닙니다.

이 차이는 논문에서 명확히 해야 합니다.

특히 similarity에는 강한 공통 구조가 있기 때문에, 네 값의 순서가 무작위일 이유가 없습니다.

따라서 저는 문장에서

> “below chance”

보다는

> “the diagonal-ranking criterion is satisfied in only 0.88% of controlled pairs, substantially below the 16.7% reference rate under uniformly random ordering”

처럼 제한하겠습니다.

---

# 5. “결정론적 역전”이라는 표현도 과하다

논문은

> “실패는 확률적 오차가 아니라 결정론적 순서 역전이다.”

라고 주장합니다. 

수식 자체에서는 맞습니다.

하지만 이 표현은 reviewer에게 불필요하게 공격받을 여지가 있습니다.

왜냐하면 \(\alpha,\beta,\gamma\) 자체가 데이터에서 추정된 값이고, measurement noise와 sample variation이 존재하기 때문입니다.

정확한 의미는

> **given the four measured similarity values, the observed ranking is algebraically determined by the inequality**

입니다.

따라서 “failure is deterministic”보다는

> “the observed ranking is algebraically implied by the measured main-effect/interactions”

가 안전합니다.

---

# 6. 플라시보 검정은 좋은데 현재 해석이 지나치다

플라시보 검정은 논문에서 꽤 좋은 실험입니다.

예를 들어 pizza 제거에 대해

* \(AUC_X=0.957\)
* \(AUC_Y=0.174\)

가 나오고, bus도 0.958 대 0.375 등으로 나타납니다. 

이는 “단순히 이미지가 망가져서 embedding이 변한 것”이라는 설명을 약화시키는 데 도움이 됩니다.

하지만 논문이

> “어떤 일반적 화질 열화도 이 부호 반전을 만들 수 없으므로”

라고 가는 순간 과도합니다. 

그건 입증하기 어렵습니다.

인페인팅은 단순한 화질 열화가 아니라 semantic/contextual 변화도 발생시키기 때문입니다.

더 중요한 문제는 저자 스스로 인정하듯 person 제거가 다른 객체까지 건드립니다. 

따라서 placebo test는

> “편집 artifact가 완전히 배제되었다”

가 아니라

> “단순한 global image degradation만으로 관찰된 object-specific effect를 설명하기 어렵다”

정도로 제한하는 것이 적절합니다.

---

# 7. 가장 큰 실험적 약점: 자연 이미지라고 해서 causal control이 완성되는 것은 아니다

논문은 BEAF의 object-removal pair를 이용합니다.

원본:

$$
I_+
$$

객체 제거:

$$
I_-
$$

라는 구조입니다.

이건 좋은 출발점이지만, 실제로는

$$
I_- = \operatorname{Inpaint}(I_+,X)
$$

입니다.

즉,

> “X만 제거된 동일 이미지”

와

> “X가 없었을 때 원래 촬영되었을 법한 이미지”

는 다릅니다.

이 문제는 논문 스스로 한계로 인정하고 있습니다. 

하지만 이 문제는 생각보다 중요합니다.

특히 \(\beta\)가 이미지 상태의 main effect이기 때문에, inpainting artifact가 systematic하게 β를 바꾼다면 γ 측정에도 영향을 줄 수 있습니다.

따라서 리뷰어는 다음을 요구할 가능성이 높습니다.

1. 실제 absence image
2. synthetic removal
3. placebo removal
4. background-preserving edit

정도의 비교입니다.

2페이지 논문이라면 전부 넣을 필요는 없지만, 적어도 **artifact robustness 하나는 본문에 있어야 한다고 봅니다.**

---

# 8. 텍스트 최소쌍도 장점과 약점이 동시에 있다

텍스트에서 같은 단어 집합을 유지하는 것은 좋은 통제입니다.

예를 들어:

> A: “The scene features a pizza, but lacks a cup.”

> B: “The scene features a cup, but lacks a pizza.”

처럼 구성하면 단순히 “not/no가 있느냐”를 탐지하는 shortcut을 제거할 수 있습니다.

논문도 이 점을 명시합니다. 

그러나 reviewer는 바로 다른 질문을 합니다.

> “그렇다면 이것은 실제 natural negation comprehension을 측정하는가?”

두 문장이 동일한 단어 집합을 공유하는 것은 오히려 매우 artificial한 linguistic construction입니다.

따라서 이 benchmark는

> **natural negation benchmark**

가 아니라

> **controlled interaction diagnostic**

이라고 명확히 규정해야 합니다.

이것은 단점이 아니라 오히려 논문의 목적과 잘 맞습니다.

---

# 9. “정렬 개입은 원리적으로 실패한다”는 주장은 현재 가장 위험하다

이 부분은 제가 리뷰어라면 가장 먼저 공격하겠습니다.

논문은 대략

$$
\gamma=0.00056
$$

이고 \(|\alpha|=0.00781\)이므로 약 14배 차이가 나며, alignment를 개선해도 γ가 최대 약 4.6배밖에 증가하지 않으므로 부족하다고 주장합니다. 

문제는 **“alignment improvement의 상한 = 4.6×”라는 논리**입니다.

특정 probe-direction alignment를 개선하는 것이 전체 similarity function에서 \(\gamma\)를 최대 4.6배만 증가시킨다는 것은 일반적인 명제가 아닙니다.

특정 intervention class에서는 계산할 수 있지만,

> “representation alignment를 개선하는 방법은 원리적으로 이 정도밖에 못 한다.”

로 일반화할 수는 없습니다.

실제로 업로드된 CVPR 2026 연구는 negation direction을 steering해서 CLIP을 개선하고 있고,  다른 연구들은 subspace modeling이나 embedding correction을 사용합니다.

따라서 이 논문은

> “all alignment-based approaches are insufficient”

가 아니라

> **“single-direction alignment interventions of the tested form cannot overcome the observed main-effect dominance under our controlled setting.”**

라고 해야 합니다.

이 차이가 상당히 큽니다.

---

# 10. 특히 W를 이용한 일반적인 linear transformation까지 막지는 못한다

이 부분은 더 근본적입니다.

만약

$$
S(u,v)=u^\top v
$$

에서

$$
S_W(u,v)=u^\top Wv
$$

를 허용한다면,

$$
\gamma_W=u^\top Wv
$$

가 됩니다.

즉 원래의 \(\gamma\)가 작다고 해서 모든 linear cross-modal transformation 이후의 \(\gamma_W\)도 작다는 결론은 나오지 않습니다.

실제로 프로젝트의 이전 분석에서도 이 문제가 이미 지적되었습니다. \(W\)를 넣으면 γ 자체가 변하기 때문에 원래 γ를 고정한 상태에서 intervention의 상한을 계산해서는 안 됩니다. 

따라서 현재 논문의 “기존 intervention의 도달 가능성” 부분은 **수학적 claim을 상당히 좁혀야 합니다.**

이건 minor issue가 아닙니다. 잘못 방어하면 논문의 핵심 결론 중 하나가 무너집니다.

---

# 11. “67.4%가 도달 범위”도 특히 조심해야 한다

초안에서는 주효과를 제거하면

$$
\alpha_W=\beta_W=0
$$

이고 따라서

$$
\Delta_W>0 \iff \gamma_W>0
$$

이므로 67.4%가 가능하다고 해석합니다. 

여기서 중요한 문제가 있습니다.

$$
\gamma_W\neq\gamma.
$$

따라서 원래 γ가 양수인 샘플의 비율을 가지고 “새로운 score function의 upper bound가 67.4%”라고 말할 수 없습니다.

\(W\)가 적용되면 γ도 함께 변하기 때문입니다.

이건 이전 분석에서 이미 정확하게 지적된 오류입니다. 

따라서 2페이지 판본에서는 이 부분을 **삭제하는 것이 오히려 논문을 강하게 만듭니다.**

---

# 12. 2페이지 논문으로서는 너무 많은 것을 주장한다

현재 논문은 동시에 다음을 주장합니다.

1. 기존 연구의 공통 전제에 대한 비판
2. 새로운 object-level controlled protocol
3. 2×2 algebraic decomposition
4. new necessary-and-sufficient condition
5. CLIP의 below-chance behavior 설명
6. γ의 statistical significance
7. object specificity
8. 기존 intervention의 한계
9. alignment의 upper bound
10. 새로운 score function의 potential

2페이지에서는 너무 많습니다.

결과적으로 각 주장에 대한 증거가 충분히 설명되지 못합니다.

저라면 **세 가지 주장만 남깁니다.**

### Claim 1

기존 probing에서 “정보가 존재한다”는 결론은 controlled object-level setting에서는 충분하지 않다.

### Claim 2

부정 matching에는 cross-modal interaction term이 필요하며, 실제 CLIP에서는 그 크기가 main effects보다 현저히 작다.

### Claim 3

따라서 CLIP의 negation failure는 단순한 representation absence가 아니라 **weak cross-modal interaction relative to strong marginal biases**로 진단할 수 있다.

이 세 개면 충분합니다.

---

# 13. 현재 논문에서 가장 강한 실험은 의외로 2×2 accuracy 자체가 아니다

가장 강한 결과는 다음의 조합입니다.

$$
\gamma >0
$$

그런데

$$
\gamma \ll |\alpha|,\ |\beta|
$$

이고,

$$
|\alpha|>\gamma
$$

가 33/33에서 나타난다는 것입니다. 

그리고 γ가 단순한 global artifact가 아니라 object-specific이라는 placebo test가 붙습니다. 

이 세 가지가 합쳐져야 합니다.

즉 논문의 논리는

> “γ가 없다.”

가 아닙니다.

오히려

> **“γ는 있다. 그런데 너무 작다.”**

입니다.

이 차이는 매우 중요합니다.

기존 연구를 반박하는 것도 아니고,

> “CLIP에 negation information이 없다.”

고 말하는 것도 아닙니다.

오히려 기존 probing 결과를 인정하면서 그것의 해석을 수정합니다.

---

# 14. 기존 연구와의 관계는 상당히 괜찮다

여기서는 논문의 위치 선정이 꽤 좋습니다.

기존 연구 중 일부는 다음과 같이 말합니다.

> negation information exists → alignment/intervention하면 된다.

예를 들어 CLIP의 내부 negation processing을 분석한 연구가 있고,  embedding-space negation direction을 직접 찾아 steering하는 연구도 있습니다. 

반면 본 논문은

> information exists ≠ information is strong enough to determine ranking

이라고 주장합니다.

이것은 **기존 연구의 반박이라기보다는 한 단계 더 세밀한 진단**입니다.

이 포지셔닝은 유지하는 게 좋습니다.

다만 “기존 연구가 틀렸다”는 식으로 쓰면 안 됩니다.

---

# 15. novelty는 현재 상태에서 어느 정도인가?

제 판단은 다음과 같습니다.

### 수학적 novelty

**낮음**

2×2 Hadamard decomposition과 interaction contrast 자체는 새롭다고 보기 어렵습니다.

### 실험 프로토콜 novelty

**중간 이상**

동일 자연 이미지에서 객체 하나를 제거한 최소쌍 + 동일 word-set 텍스트 쌍을 결합하여 cross-modal interaction을 직접 측정하는 것은 꽤 괜찮습니다.

### empirical finding novelty

**중간~높음**

“negation information이 존재한다”에서 멈추지 않고

$$
|\alpha| \gg \gamma
$$

라는 **상대적 magnitude**를 보여준 것이 핵심입니다.

### theoretical novelty

**낮음~중간**

새로운 theorem이라기보다는 기존 factorial decomposition을 VLM diagnosis에 적용한 것입니다.

### 전체 novelty

**중간 정도**

논문으로 만들 수 있는 novelty는 있습니다. 하지만 제목과 abstract에서 “new algebraic theory”처럼 보이게 만들면 오히려 reject될 가능성이 높아집니다.

---

# 16. 제가 실제 리뷰어라면 이렇게 평가하겠습니다

| 항목                | 평가                           |
| ----------------- | ---------------------------- |
| 문제 중요성            | 4/5                          |
| 실험 아이디어           | 4/5                          |
| 수학적 정확성           | 4/5                          |
| 수학적 novelty       | 2/5                          |
| empirical novelty | 3.5/5                        |
| 실험 통제             | 3/5                          |
| 기존 연구와의 차별성       | 3/5                          |
| 주장과 증거의 일치        | 2.5/5                        |
| 재현 가능성            | 4/5                          |
| 전체                | **Weak Reject / Borderline** |

가장 큰 이유는 **결과가 약해서가 아니라 claim이 evidence보다 조금씩 강하기 때문**입니다.

---

# 17. 리뷰어 코멘트를 실제로 쓴다면

> **Summary:**
> This paper studies negation failures in CLIP through a controlled 2×2 matching setup, decomposing image-text similarities into image and text main effects and a cross-modal interaction term. The authors show that the interaction term associated with negation is present but substantially weaker than the marginal effects, and argue that this explains the poor performance of CLIP under controlled negation matching.

> **Strengths:**
> The controlled minimal-pair construction is a useful improvement over evaluations where image-level confounds may allow probing methods to exploit scene-level differences. The decomposition provides an interpretable diagnostic of the relative contribution of marginal and interaction effects. The empirical finding that the interaction term is consistently much smaller than the text main effect is potentially interesting and provides a useful refinement to prior claims that negation information is already represented.

> **Major concern 1:**
> The main algebraic proposition appears largely to be a reparameterization of a 2×2 matrix using the standard interaction basis. While correct, its novelty is limited. The paper should therefore emphasize the empirical diagnostic and not present the proposition itself as a major theoretical contribution.

> **Major concern 2:**
> Several claims about the fundamental limitations of alignment-based interventions appear stronger than what is established experimentally. In particular, improving the alignment of probe directions does not generally imply a bounded improvement of the interaction coefficient γ, since a general cross-modal transformation can alter γ itself. The intervention analysis should be restricted to clearly specified transformation classes.

> **Major concern 3:**
> The object-removal pairs are generated through inpainting, which may introduce systematic visual artifacts or contextual changes. The placebo analysis reduces but does not completely eliminate this concern. Additional robustness experiments or more cautious causal language are needed.

> **Major concern 4:**
> The 16.7% reference rate corresponds to uniformly random ordering of four scores, not necessarily to a statistical chance level for the actual CLIP similarity distribution. The interpretation of the 0.88% result should therefore be more carefully qualified.

> **Overall assessment:**
> The paper contains a potentially useful empirical diagnosis, but its novelty and claims should be reframed around the key observation that representational availability does not imply sufficiently strong cross-modal interaction for ranking decisions. With a narrower theoretical claim and stronger control of image-editing artifacts, the work could become substantially more convincing.

---

## 최종적으로 제가 논문을 고친다면

가장 중요한 것은 **새 실험을 무작정 추가하는 것이 아니라 claim을 줄이는 것**입니다.

현재:

> “기존 방법들은 원리적으로 이 문제를 해결할 수 없다.”

보다

> **“기존의 representation-level evidence는 negation matching에 필요한 cross-modal interaction의 충분한 크기를 보장하지 않는다.”**

가 훨씬 강합니다.

그리고 논문의 한 줄짜리 핵심은 저는 다음으로 잡겠습니다.

> **CLIP does encode object-level negation information, but the corresponding cross-modal interaction is too weak relative to its marginal similarity biases to control retrieval rankings.**

이렇게 가면 기존의 NegBench, Quantmeyer, Sammani, SpaceVLM, ICLR 2026의 embedding correction 계열을 정면으로 부정하지 않으면서도, 그 연구들이 주로 묻지 않았던 **“representation에 존재하는 신호의 크기가 실제 decision을 지배할 만큼 충분한가?”**라는 질문을 가져올 수 있습니다. 기존 연구들이 negation data, embedding direction, subspace, token/feature intervention 등 서로 다른 해결책을 제시하고 있다는 점도 오히려 이 framing을 뒷받침합니다.

**결론적으로, 저는 이 연구를 폐기할 정도로 약하다고 보지는 않습니다. 다만 현재의 2페이지 버전은 “수학적 명제 + 여러 intervention의 불가능성”을 너무 앞세우고 있습니다. 논문의 진짜 기여인 `controlled measurement → γ exists → γ is much smaller than marginal effects → ranking failure`의 흐름으로 재구성하는 것이 훨씬 설득력 있습니다.**

### 논문 요약 (Summary)

본 논문(PAPER_2PAGE)은 CLIP 계열 듀얼 인코더 모델이 객체의 부재나 부정(Negation) 질의에서 반복적으로 실패하는 원인을 수식적으로 규명하고, 정보의 표상(Representation)과 유사도 상호작용(Similarity Interaction) 간의 괴리를 분해(Decomposition) 이론을 통해 정량화합니다.

저자들은 BEAF 기반 인페인팅 최소쌍 이미지와 어휘 집합이 통제된 AB-swap 텍스트 쌍으로 구성된 $2 \times 2$ 매칭 행렬 $S_{ab}$를 전체 평균 $C$, 이미지 주효과 $\beta$, 텍스트 주효과 $\alpha$, 교차항 $\gamma$로 분해합니다. 이를 통해 Winoground의 Group Score 판정과 동치인 부정 검색 성공 조건이 $\gamma > \max(\vert{}\alpha\vert{}, \vert{}\beta\vert{})$임을 대수적으로 증명합니다.

실험 결과, 선형 프로브를 통해 객체 존재/부정 정보가 인코더에 유의미하게 표상되어 있음(이미지 AUC 0.759, 텍스트 쌍별 정확도 64.69%)에도 불구하고, 실제 교차항 $\gamma$(0.00121)가 주효과 $\beta$(0.00628)의 1/5.2에 불과하여 $2 \times 2$ 정답률이 4.03%(우연 확률 16.67%)로 붕괴함을 보입니다. 또한, 학습 없는 rank-1 bilinear 헤드 $v^T (w_I w_T^T) t$를 적용할 경우 코사인 유사도 대비 6배 높은 24.96%의 정답률에 도달함을 보임으로써 병목이 인코더의 표상 능력이 아닌 유사도 매칭 인터페이스(항등 행렬 고정)에 있음을 논증합니다.

---

### 주요 강점 (Strengths)

* **Winoground 평가 지표의 엄밀한 대수적 환원:**
Winoground(Thrush et al., 2022)가 제기했던 '시각-언어 모델의 합성적 추론 실패(우연 확률 이하의 Group Score)' 현상을 모호한 '융합 실패(Fusion Failure)' 가설에 머무르지 않고, $S_{ab} = C + a\beta + b\alpha + ab\gamma$라는 좌표 변환과 $\Delta(S) = 2\gamma - 2\max(\vert{}\alpha\vert{}, \vert{}\beta\vert{})$라는 항등식을 통해 완벽히 수학적으로 정식화했습니다. Group Score의 성공과 실패를 결정론적 부등식 조건으로 명쾌하게 규명한 이론적 기여가 뛰어납니다.


* **선행 연구 간의 대립적 시각 통합:**
Alshehri et al.(2026)의 '유사도의 개념 증거 평균 풀링' 주장과 Koishigarina et al.(2026)의 '단일 모달리티 내 결합 정보 보존' 주장, 그리고 NegBench(Alhamoud et al., 2025)의 '긍정 편향(Affirmation Bias)'을 대립되는 개념이 아닌 동일한 분해식의 서로 다른 항($\beta$ 지배, $\gamma \neq 0$, $\alpha > 0$)으로 통합하여 해석 틀을 제공합니다.


* **엄격한 실험적 통제 (AB-swap & BEAF 최소쌍):**
단순히 "not a dog"과 같은 템플릿 기반 부정어를 비교할 때 발생하는 Bag-of-Words 편향을 배제하기 위해, 동일한 단어 구성을 위치/결합만 바꾼 AB-swap 텍스트와 배경이 동일 통제된 인페인팅 이미지를 사용하여 모델의 구조적 취약점을 정밀 타격했습니다.


* **경량 rank-1 Bilinear 솔루션의 실험적 검증:**
코사인 유사도(항등 행렬)의 한계를 지적하는 것에 그치지 않고, 두 단봉 프로브 법선의 외적 $v^T (w_I w_T^T) t$만으로 추가 학습 없이 성능이 4.03%에서 24.96%로 6배 도약함을 입증하여 실질적인 대안 방향을 제시했습니다.



---

### 비판적 검토 및 한계점 (Weaknesses & Critical Inquiries)

* **Open-Vocabulary / Zero-Shot 환경에서의 확장성(Scalability) 한계:**
제시된 rank-1 bilinear 채점기나 2-bit 채점 함수 $f(\hat{a}, \hat{b})$는 특정 개념(Concept-specific)에 대해 사전 학습된 선형 프로브의 법선 벡터 $w_I, w_T$를 요구합니다. 그러나 CLIP의 본질적인 강점은 미지의 임의 질의에 대응하는 Zero-shot Retrieval입니다. 사전에 정의되지 않은 열린 어휘(Open-vocabulary) 환경에서 개념별 프로브 가중치 없이 어떻게 이 rank-1 상호작용을 일반화할 것인지에 대한 구체적인 메커니즘 제시가 부족합니다.


* **부정 유형의 협소성 (객체 부재에 국한):**
본 연구는 BEAF 인페인팅 데이터셋의 특성상 '다객체 장면에서의 객체 존재/부재'만을 다루고 있습니다. 그러나 Kang et al.(2025)이나 NegBench, Winoground에서 다루는 부정 현상은 속성 부정(e.g., "pizza without green peppers", "orange without leaves"), 관계 부정, 행위 부정(e.g., "horse that is not urinating") 등 복잡한 양상을 띱니다. 본 논문의 요인 분해 프레임워크가 속성 바인딩이나 관계적 부정에도 동일한 $\gamma / \max(\vert{}\alpha\vert{}, \vert{}\beta\vert{})$ 비율로 일반화되는지 검증이 필요합니다.


* **듀얼 인코더 외 아키텍처 및 복합 구조에 대한 분석 부재:**
Winoground 논문에서는 듀얼 인코더(CLIP, FLAVA Contrastive)뿐만 아니라 Cross-attention을 사용하는 Dual-stream(LXMERT, ViLBERT) 및 Single-stream(UNITER, VinVL) 트랜스포머도 함께 평가했습니다. 본 논문은 듀얼 인코더의 내적 연산 구조를 주된 실패 원인(코사인은 rank-1을 항등 행렬로 고정)으로 지목하지만, Cross-attention 융합 레이어를 가진 모델들조차 Winoground Group Score에서 무작위 확률 이하로 실패하는 현상을 이 분해 이론이 어떻게 설명할 수 있는지 논의가 확장되어야 합니다.


* **텍스트 위치 편향(Positional Bias) 통제의 한계:**
섹션 3.1에서 어순 제어를 위해 위치 교차 평가를 수행했을 때 정확도가 73.80%에서 64.69%로 12% 하락함을 인정하고 있습니다. 이는 텍스트 인코더가 학습한 정보의 상당 부분이 순수한 의미적 극성(Polarity)이 아닌 단순 어순 휴리스틱(Surface order)에 의존하고 있을 위험을 시사합니다.



---

### 완결 논문(`PAPER.md`)을 위한 제언 (Suggestions for Improvement)

1. **Test-Time Matching(TTM) 및 벤치마크 착시 비판 강화:**
최근 Zhu et al.(2026) 등에서 제안된 GroupMatch($S_{11} + S_{22} > S_{12} + S_{21}$)는 대수적으로 $\alpha, \beta$를 강제 소거하여 $\gamma > 0$만을 평가하는 특수 케이스임을 보다 명시적으로 강조할 필요가 있습니다. $2 \times 2$ 평가 환경과 실제 1:N 갤러리 검색($\beta$가 소거되지 않는 환경)의 괴리를 본 논문의 분해식으로 설명하면 논문의 파급력이 훨씬 커질 것입니다.


2. **미세조정 모델(NegCLIP, ConCLIP 등)의 기하학적 왜곡 분석 수록:**
NegBench 등에서 합성 부정 데이터로 파인튜닝한 모델들(NegCLIP, ConCLIP)이 왜 여전히 근본적인 해결책이 되지 못하는지, 즉 $\gamma$를 키우지 못하고 $\alpha$의 부호만 음수로 과도하게 밀어버리는 현상을 분해 계수 그래프와 함께 실증 데이터로 제시하면 완성도가 크게 향상될 것입니다.


3. **일반화 가능한 아키텍처 제언 추가:**
개념별 프로브에 의존하는 rank-1 헤드를 넘어, 텍스트 질의로부터 동적으로 극성 축을 추론하거나(e.g., NEGTOME의 토큰 병합 및 극성 부스팅 방식, 혹은 SpaceVLM의 서브스페이스 모델링), cross-modal projection layer를 학습하는 일반화된 추론 아키텍처 방향성을 결론부에서 구체화할 것을 권장합니다.



---

### 종합 평가

**판정:** **Strong Accept (국내 학술대회 2페이지 기준 최우수 수준, 확장 저널/국제학술대회 투고 강력 권장)**

본 논문은 직관과 휴리스틱에 의존하던 V&L 모델의 부정 매칭 실패 문제를 명쾌한 대수적 분해식으로 정식화하고 실증적으로 증명한 매우 우수한 연구입니다. 압축적인 2페이지 분량 내에 문제 제기, 수학적 동치 증명, 통제된 데이터셋 구성, 프로빙 및 rank-1 대안 제시까지 논리적 흐름이 매우 탄탄합니다. 상기 지적된 열린 어휘 환경에서의 일반화 방안 및 다양한 부정 유형에 대한 논의가 보강된다면 최우수 국제 인공지능 학술대회(CVPR/ICLR/NeurIPS) 본회의 논문으로도 손색이 없을 것으로 판단됩니다.

# 논문 비판적 리뷰

## 1. 총평

이 논문은 CLIP 계열 모델의 부정 매칭 실패를 단순히 “부정 정보를 표현하지 못하기 때문”이라고 설명하는 대신, **정보가 임베딩에 존재하더라도 코사인 유사도에서 충분히 활용되지 않을 수 있다**는 문제로 재구성한다. 특히 2×2 매칭 점수를 주효과와 교차항으로 분해하고, 부정 검색의 성공 조건을 \(\gamma > \max(|\alpha|,|\beta|)\)로 제시한 점은 논문의 핵심 아이디어이자 가장 강한 부분이다.  [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/61333062/4c4ba4d0-8c5d-4fb4-8f4d-12b0f128db58/PAPER_2PAGE.pdf?AWSAccessKeyId=ASIA2F3EMEYE4KAQ7JI2&Signature=tXGsMX339QK7pQebh7nQ3VvrYrM%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEM7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIFR25qlXD0mxPyNT01izBIY9WJYsbSL2EgHfZAk8qOxDAiBH4Jo9vl%2BoaknN%2Fxp94KuTg3%2FDj9I2Fgy8k10kXFQlsyr8BAiX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIM38aoc6LzpKb2TgyhKtAESW2RHQyb1TiAChs%2BIc1ECwW%2BlyLxmf0TmaAOeApr17T27P%2BxpVQgDx%2BQhEppJuU%2FrhsnzhZHaGDFLTCbZfe6C822B8NV1cpJQnRISaHc7QDbn6KruMLHRRJ7dWrbb%2FuCIttp0PMyrj4h8x7kXg10gvCDW8AXiuTD2brmhY8x3RET6E6ncie2HUxce3APk4wXY1nl6Pv9%2FuurjeLScepFYa1KU%2BAC7GzSYXFBX7NLznJVCaBbeKrUrRVNcd26gLqv1Bu5zx9gYmB2akIgfobeS%2BNDiWiSfcHXduxqK1LgNVa%2Fm%2Fpuzhww1Z3Is4Y2zE3CjUH%2FAbN5LW0QFOC8OSJj0kZWODbRfFTtwcueEp4klzDMEzhO3%2FszO%2FZR%2FmknUJvFm005kW6x%2Fb3hiEHWW8t8pbVoWTWyzxJd7gm%2BZLeE9MDCqd1Yj%2FSVmfUi0ePEoguwFk%2BNdxvt9PRuHk%2BW8eE1sPkfI5CpElCdQeMWDpvNxiFH%2Fo7%2FDzStwyD1%2BasSZpkd3MzmL1YSnmzAR%2FA9Ae1tA4q6jqlZs50AV16AyQd1rcWFdGTW4a3I2Kt96M4tFAyyJBkfpZ1FPSVP1qLWpNBCMJiLULHswa%2FlE0ZkWsnFGaa7%2BWv33z%2FUAl2TxXFtKQO8N5ziTreolMXfSqsdCyNCSuEGXuLbN%2BXihL%2Bt8GE8MqJmmrfQdNqi1my%2Bp8O%2FbRQysETEwFbhkV%2BXI%2BBV4r7IP6rcIAHwyCLsF%2B7m%2F%2BBEYGUxx8NLlPJAP9TMBvboOa4kAcLzyZetiws486l90y59iTClkNbUBjqZAQSFAuGKOpO6i0OmA79iJpdqq7%2BTpTM0pzwWkyqrAt5ybK%2FNBunE8pckT6uAJ8X29wDNdtTKJ0iTch9FbMv9wn1wd7EGCkUnVvUCmQDreUxa2cf%2Bs%2FtIJW%2F8H0q24ybivo8k7VCgBcysO6IWzXVb0oCiCDmojmjb1nbMeLCmWBa1gTZU7PA6HAx67wPhqgHMI34K3L977nXgIw%3D%3D&Expires=1788188152)

다만 현재의 2페이지 판본만을 기준으로 보면, 이론적 분해 자체는 거의 정의에 가까운 항등식인 반면, 그 분해가 CLIP의 일반적인 실패 원인이라는 경험적 주장을 뒷받침하는 실험 설계와 통계적 검증은 아직 충분히 제시되지 않았다. 특히 데이터 구성, 모델 선택, 프로브 검증, 통계 단위, 기준선, 재현성 정보가 제한되어 있어, **흥미로운 진단 가설로서는 강하지만 일반적 메커니즘에 대한 확정적 결론으로는 과도하다**고 평가한다.

현재 형태라면 학술대회 리뷰 기준으로는 **Weak Reject 또는 Major Revision**에 가깝다. 완결 논문에 언급된 아홉 모델 비교, 개입 실험, 외부 벤치마크 검정이 충분히 공개되어 있고 방법론적 결함을 보완한다면 평가가 올라갈 수 있다.

## 2. 주요 강점

### 2.1 문제 설정이 명확하다

논문은 “표현되어 있음”과 “유사도 순위를 결정함”을 구별한다. 이는 멀티모달 모델 분석에서 자주 혼동되는 두 수준을 분리한다는 점에서 중요하다.

- 프로브가 객체 존재나 부정 관련 정보를 예측할 수 있다는 것.
- 원래의 코사인 유사도가 올바른 이미지-문장 조합을 선택한다는 것.

두 조건은 논리적으로 동일하지 않으며, 논문은 이 간극을 \(\alpha,\beta,\gamma\)라는 동일한 2×2 구조 안에서 분석하려 한다. 이 문제의식은 직관적이고, 기존의 “정보가 없다” 대 “정보는 있지만 활용하지 못한다”는 논쟁을 더 정교하게 다룰 가능성이 있다.

### 2.2 2×2 분해가 해석 가능하다

\[
S_{ab}=C+a\beta+b\alpha+ab\gamma
\]

라는 표현은 네 개의 점수를 평균, 이미지 주효과, 텍스트 주효과, 상호작용 효과로 나누는 표준적인 요인 설계와 유사하다. 이 분해는 각 항의 의미를 직관적으로 제공한다.

- \(\alpha\): 텍스트 상태에 따른 평균적 유사도 차이.
- \(\beta\): 이미지 상태에 따른 평균적 유사도 차이.
- \(\gamma\): 이미지와 텍스트 상태가 올바르게 결합될 때 발생하는 상호작용.

또한 논문은 정답 마진을

\[
\Delta(S)=\min(S_{++},S_{--})-\max(S_{+-},S_{-+})
\]

로 정의하고, 다음을 제시한다.

\[
\Delta(S)=2\gamma-2\max(|\alpha|,|\beta|)
\]

이 식이 정확하다면, 2×2 정답 조건이 \(\gamma > \max(|\alpha|,|\beta|)\)로 환원된다는 주장은 깔끔하다. 경험적으로 보고한 판정 결과와 대수적 판정 결과가 거의 완전히 일치한다는 점도, 적어도 구현 오류를 점검했다는 측면에서는 긍정적이다.  [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/61333062/4c4ba4d0-8c5d-4fb4-8f4d-12b0f128db58/PAPER_2PAGE.pdf?AWSAccessKeyId=ASIA2F3EMEYE4KAQ7JI2&Signature=tXGsMX339QK7pQebh7nQ3VvrYrM%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEM7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIFR25qlXD0mxPyNT01izBIY9WJYsbSL2EgHfZAk8qOxDAiBH4Jo9vl%2BoaknN%2Fxp94KuTg3%2FDj9I2Fgy8k10kXFQlsyr8BAiX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIM38aoc6LzpKb2TgyhKtAESW2RHQyb1TiAChs%2BIc1ECwW%2BlyLxmf0TmaAOeApr17T27P%2BxpVQgDx%2BQhEppJuU%2FrhsnzhZHaGDFLTCbZfe6C822B8NV1cpJQnRISaHc7QDbn6KruMLHRRJ7dWrbb%2FuCIttp0PMyrj4h8x7kXg10gvCDW8AXiuTD2brmhY8x3RET6E6ncie2HUxce3APk4wXY1nl6Pv9%2FuurjeLScepFYa1KU%2BAC7GzSYXFBX7NLznJVCaBbeKrUrRVNcd26gLqv1Bu5zx9gYmB2akIgfobeS%2BNDiWiSfcHXduxqK1LgNVa%2Fm%2Fpuzhww1Z3Is4Y2zE3CjUH%2FAbN5LW0QFOC8OSJj0kZWODbRfFTtwcueEp4klzDMEzhO3%2FszO%2FZR%2FmknUJvFm005kW6x%2Fb3hiEHWW8t8pbVoWTWyzxJd7gm%2BZLeE9MDCqd1Yj%2FSVmfUi0ePEoguwFk%2BNdxvt9PRuHk%2BW8eE1sPkfI5CpElCdQeMWDpvNxiFH%2Fo7%2FDzStwyD1%2BasSZpkd3MzmL1YSnmzAR%2FA9Ae1tA4q6jqlZs50AV16AyQd1rcWFdGTW4a3I2Kt96M4tFAyyJBkfpZ1FPSVP1qLWpNBCMJiLULHswa%2FlE0ZkWsnFGaa7%2BWv33z%2FUAl2TxXFtKQO8N5ziTreolMXfSqsdCyNCSuEGXuLbN%2BXihL%2Bt8GE8MqJmmrfQdNqi1my%2Bp8O%2FbRQysETEwFbhkV%2BXI%2BBV4r7IP6rcIAHwyCLsF%2B7m%2F%2BBEYGUxx8NLlPJAP9TMBvboOa4kAcLzyZetiws486l90y59iTClkNbUBjqZAQSFAuGKOpO6i0OmA79iJpdqq7%2BTpTM0pzwWkyqrAt5ybK%2FNBunE8pckT6uAJ8X29wDNdtTKJ0iTch9FbMv9wn1wd7EGCkUnVvUCmQDreUxa2cf%2Bs%2FtIJW%2F8H0q24ybivo8k7VCgBcysO6IWzXVb0oCiCDmojmjb1nbMeLCmWBa1gTZU7PA6HAx67wPhqgHMI34K3L977nXgIw%3D%3D&Expires=1788188152)

### 2.3 AB-swap 설계는 표면적 부정 탐지를 견제한다

두 텍스트가 거의 동일한 단어 집합을 공유하고 결합만 바꾸는 AB-swap 구성은, 단순한 “not” 토큰 탐지나 부정 표지 유무 탐지를 줄이려는 시도다. 다음 예시는 그 목적을 잘 보여준다.

- “The scene features a pizza, but lacks a cup.”
- “The scene features a cup, but lacks a pizza.”

이러한 설계는 일반적인 긍정 문장과 부정 문장을 비교하는 것보다 더 엄격한 테스트가 될 수 있다. 위치 성분을 통제했을 때 텍스트 프로브 성능이 감소하지만 여전히 우연 수준을 크게 넘는다는 분석도 문제의식에 부합한다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/61333062/4c4ba4d0-8c5d-4fb4-8f4d-12b0f128db58/PAPER_2PAGE.pdf?AWSAccessKeyId=ASIA2F3EMEYE4KAQ7JI2&Signature=tXGsMX339QK7pQebh7nQ3VvrYrM%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEM7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIFR25qlXD0mxPyNT01izBIY9WJYsbSL2EgHfZAk8qOxDAiBH4Jo9vl%2BoaknN%2Fxp94KuTg3%2FDj9I2Fgy8k10kXFQlsyr8BAiX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIM38aoc6LzpKb2TgyhKtAESW2RHQyb1TiAChs%2BIc1ECwW%2BlyLxmf0TmaAOeApr17T27P%2BxpVQgDx%2BQhEppJuU%2FrhsnzhZHaGDFLTCbZfe6C822B8NV1cpJQnRISaHc7QDbn6KruMLHRRJ7dWrbb%2FuCIttp0PMyrj4h8x7kXg10gvCDW8AXiuTD2brmhY8x3RET6E6ncie2HUxce3APk4wXY1nl6Pv9%2FuurjeLScepFYa1KU%2BAC7GzSYXFBX7NLznJVCaBbeKrUrRVNcd26gLqv1Bu5zx9gYmB2akIgfobeS%2BNDiWiSfcHXduxqK1LgNVa%2Fm%2Fpuzhww1Z3Is4Y2zE3CjUH%2FAbN5LW0QFOC8OSJj0kZWODbRfFTtwcueEp4klzDMEzhO3%2FszO%2FZR%2FmknUJvFm005kW6x%2Fb3hiEHWW8t8pbVoWTWyzxJd7gm%2BZLeE9MDCqd1Yj%2FSVmfUi0ePEoguwFk%2BNdxvt9PRuHk%2BW8eE1sPkfI5CpElCdQeMWDpvNxiFH%2Fo7%2FDzStwyD1%2BasSZpkd3MzmL1YSnmzAR%2FA9Ae1tA4q6jqlZs50AV16AyQd1rcWFdGTW4a3I2Kt96M4tFAyyJBkfpZ1FPSVP1qLWpNBCMJiLULHswa%2FlE0ZkWsnFGaa7%2BWv33z%2FUAl2TxXFtKQO8N5ziTreolMXfSqsdCyNCSuEGXuLbN%2BXihL%2Bt8GE8MqJmmrfQdNqi1my%2Bp8O%2FbRQysETEwFbhkV%2BXI%2BBV4r7IP6rcIAHwyCLsF%2B7m%2F%2BBEYGUxx8NLlPJAP9TMBvboOa4kAcLzyZetiws486l90y59iTClkNbUBjqZAQSFAuGKOpO6i0OmA79iJpdqq7%2BTpTM0pzwWkyqrAt5ybK%2FNBunE8pckT6uAJ8X29wDNdtTKJ0iTch9FbMv9wn1wd7EGCkUnVvUCmQDreUxa2cf%2Bs%2FtIJW%2F8H0q24ybivo8k7VCgBcysO6IWzXVb0oCiCDmojmjb1nbMeLCmWBa1gTZU7PA6HAx67wPhqgHMI34K3L977nXgIw%3D%3D&Expires=1788188152)

### 2.4 코사인 이외의 읽기 방식에 대한 분석이 흥미롭다

동일한 임베딩을 프로브 기반 규칙이나 rank-1 bilinear 헤드로 읽었을 때 성능이 개선된다는 결과는 논문의 핵심 주장을 구체화한다. 즉, 표현 자체를 바꾸지 않고도 상호작용을 읽는 함수를 바꾸면 성능이 변할 수 있다는 점이다.

특히

\[
(w_I^\top v)(w_T^\top t)
=
v^\top(w_Iw_T^\top)t
\]

라는 관찰은 “필요한 구조가 고차원적이고 복잡하다”는 해석에 대한 반례가 될 수 있다. 필요한 상호작용이 rank-1 형태로도 어느 정도 포착된다면, 문제는 표현 용량보다 **고정된 유사도 함수의 귀납 편향**일 수 있다.

## 3. 주요 우려 사항

### 3.1 핵심 명제가 새롭다기보다 정의적이다

논문의 가장 중심적인 명제는 2×2 점수 배열에 대한 좌표 변환에서 직접 도출된다. 따라서 수학적으로는 정확하지만, 그 자체만으로 CLIP의 실패 원인에 대한 강한 이론적 기여라고 보기는 어렵다.

네 개의 점수에 대해 주효과와 상호작용항을 정의하면, 특정 순위 조건이 그 항들의 부등식으로 표현되는 것은 자연스럽다. 따라서 논문의 진짜 기여는 명제 1 자체가 아니라 다음 질문에 달려 있다.

- 이 분해가 기존의 compositionality 또는 negation 분석보다 실질적으로 더 많은 것을 설명하는가?
- \(\gamma\)가 실제로 인과적 병목인가?
- \(\gamma\)를 키우는 개입이 다른 데이터셋과 모델에서도 일관되게 성공하는가?
- 이 분석이 단순히 Winoground식 실패를 다른 기호로 다시 표현한 것은 아닌가?

논문은 후반부에서 \(\gamma\)를 병목으로 해석하지만, 현재 제시된 결과는 주로 상관관계와 사후적 분해에 머문다. 예를 들어 \(\gamma\)가 작은 모델에 개입하여 \(\gamma\)를 증가시키고 실제 검색 성능이 개선되는지 보여줘야 “병목”이라는 인과적 표현을 정당화할 수 있다.

### 3.2 2×2 구조가 지나치게 제한적이다

현재 분석은 다음 조건을 전제로 한다.

- 객체 하나의 존재/부재.
- 두 문장 간의 AB-swap.
- 하나의 이미지 상태와 하나의 텍스트 상태.
- 정확히 네 가지 조합으로 환원되는 매칭 문제.

이 설정은 분석 가능성을 높이지만, 실제 부정 이해의 범위를 매우 좁힌다. 논문도 관계 부정과 행위 부정을 한계로 언급하지만, 이 한계가 결론의 범위를 크게 제한한다.

예를 들어 다음 현상은 현재 분해만으로 직접 다루기 어렵다.

- “개가 공을 쫓지 않는다”와 “개가 공을 쫓는다”의 차이.
- 여러 객체 중 특정 객체에만 부정이 적용되는 경우.
- “모든”, “아무도”, “오직”, “제외하고”와 같은 양화 표현.
- 객체가 원래부터 없는 장면과 인페인팅으로 제거된 장면의 차이.
- 이미지에 객체가 있지만 가려져 있는 경우.
- 부정이 문장 전체가 아니라 특정 술어 또는 관계에 적용되는 경우.

따라서 제목과 결론에서 “CLIP의 부정 매칭 실패”라고 일반화하기보다는, **다객체 장면에서의 객체 존재 부정에 대한 2×2 매칭 실패**라고 범위를 명시하는 편이 타당하다.

### 3.3 BEAF 기반 반사실 이미지의 타당성 검증이 부족하다

BEAF의 인페인팅 기반 전후 쌍은 통제된 실험에 유용하지만, 객체 제거가 완전히 의미론적으로 중립적이라고 보기는 어렵다. 제거된 객체의 흔적, 비정상적인 배경, 경계 아티팩트, 장면의 비자연스러움이 이미지 임베딩에 영향을 줄 수 있다.

따라서 다음 비교가 필요하다.

- 원본-인페인팅 쌍.
- 자연적으로 객체가 존재하거나 부재하는 이미지 쌍.
- 객체를 제거하지 않고 가리는 쌍.
- 객체를 다른 객체로 교체한 쌍.
- 인페인팅 방식이 다른 쌍.
- 이미지 생성 모델 또는 사람에 의한 편집 결과.

현재 논문은 배경과 나머지 객체가 공유된다는 점을 강조하지만, 공유되지 않는 픽셀 변화가 어떤 방향으로 임베딩을 이동시키는지는 제시하지 않는다. \(\beta\)가 큰 이유가 객체 부재 정보 때문인지, 인페인팅 아티팩트나 장면 자연스러움 변화 때문인지 분리해야 한다.

### 3.4 프로브 결과가 “정보의 존재”를 충분히 입증하지 못한다

선형 프로브 성능이 우연보다 높다는 것은 해당 정보를 선형적으로 예측할 수 있다는 증거다. 그러나 이것만으로 모델이 의미론적 객체 부재를 표현한다고 단정하기는 어렵다.

프로브는 다음과 같은 비의미론적 단서를 사용할 수 있다.

- 이미지 편집 아티팩트.
- 객체 위치나 크기 변화.
- 특정 개념에 특유한 배경.
- 텍스트 템플릿과 어순.
- 문장 길이 또는 구두점.
- 개념별 데이터 수와 분포 차이.

논문은 텍스트 위치 성분을 일부 통제하지만, 이미지 쪽에 대해서는 유사한 통제 실험이 충분히 보이지 않는다. 특히 존재 탐지 AUC 0.759가 객체 개념을 학습한 것인지, “제거된 이미지 스타일”을 감지한 것인지 확인하려면 다음이 필요하다.

- 새로운 인페인팅 방법에 대한 전이.
- 새로운 장면 분포에 대한 전이.
- 객체 위치를 통제한 평가.
- 편집 흔적만으로는 분류할 수 없는 자연 이미지 평가.
- 개념을 분리한 훈련/검증 분할.
- 프로브의 regularization과 하이퍼파라미터 공개.

### 3.5 데이터 누수와 분할 방식이 불명확하다

논문은 42개 개념, 2,480쌍, 개념 내 5-fold 검증을 언급한다. 그러나 다음 정보가 빠져 있다.

- 동일한 원본 이미지 또는 장면이 여러 분할에 들어가는지.
- 같은 객체가 다른 문장 템플릿에서 반복되는지.
- 이미지와 텍스트의 개념이 훈련과 평가에서 어떻게 분리되는지.
- 프로브 학습, 채점 함수 선택, 최종 평가가 정확히 어떤 계층에서 분리되는지.
- 모델 선택이나 결과 보고 과정에서 테스트셋을 반복적으로 사용했는지.

특히 개념 내 5-fold는 동일 개념의 매우 유사한 이미지와 문장을 양쪽 분할에 남길 가능성이 있다. 이는 일반화 성능을 높여 보일 수 있다. 최소한 다음 세 가지 분할을 함께 보고해야 한다.

| 분할 | 검증하려는 능력 |
|---|---|
| 이미지 수준 분할 | 새로운 이미지에 대한 일반화 |
| 장면/원본 그룹 수준 분할 | 거의 동일한 장면 변형 누수 방지 |
| 개념 수준 분할 | 새로운 객체 개념으로의 일반화 |

### 3.6 통계 단위와 유의성 해석이 불충분하다

논문은 42개 개념에서 \(\gamma\)가 모두 양수이고 Wilcoxon \(p=2.3\times10^{-13}\)이라고 보고한다. 그러나 통계 검정의 독립 단위가 무엇인지 명확하지 않다. 2,480개의 쌍을 독립 표본처럼 처리했다면, 같은 개념과 장면에서 나온 관측치 간 상관을 무시했을 가능성이 있다.

또한 다음을 보고해야 한다.

- 개념별 효과 크기와 신뢰구간.
- 개념별 표본 수.
- 쌍 단위와 개념 단위의 결과 차이.
- 계층적 부트스트랩 또는 혼합효과 모델.
- 여러 모델과 여러 지표에 대한 다중비교 보정.
- 4.03% 성능의 정확한 신뢰구간.
- 16.67% 기준선이 이 데이터 생성 방식에서 정말 적절한지에 대한 설명.

“무작위 순서일 때 성공 확률이 16.67%”라는 계산은 네 점수의 순위가 균등하게 분포한다는 조건에서만 성립한다. 실제로는 점수 간 동률, 점수 상관, 주효과 구조, 데이터 구성에 따라 기준선이 달라질 수 있다. 따라서 단순한 순열 기준선과 함께 실제 라벨 셔플 실험을 제시하는 편이 더 설득력 있다.

### 3.7 수치 보고에 잠재적인 불일치가 있다

본문의 일부 수치는 정의와 집계 수준이 명확하지 않다.

- \(|\alpha|=0.00380\), \(|\beta|=0.00628\), \(\gamma=0.00121\)이 평균인지 중앙값인지 불명확하다.
- “교차항은 지배 주효과의 1/5.2”는 평균 또는 중앙값의 비율인지 확인이 필요하다.
- \(\gamma/\max(|\alpha|,|\beta|)\)의 중앙값 0.1828과 앞의 요약 통계가 어떤 관계인지 설명이 부족하다.
- 2×2 정답률 4.03%가 개념별 평균인지, 전체 쌍을 합친 micro average인지 알기 어렵다.
- “378개 중 368개”와 “378개 중 0개”는 모델-개념 조합 단위인데, 각 조합의 표본 수가 동일한지 제시되지 않았다.

이런 수치는 논문의 결론을 직접 지지하므로, 표와 정의를 통해 집계 방식을 명시해야 한다.

### 3.8 rank-1 헤드에 대한 주장이 과장되어 있다

프로브 출력의 곱이 rank-1 bilinear 함수라는 점은 맞지만, 이것이 곧 “필요한 것은 rank 1”이라는 결론을 의미하지는 않는다. 현재 결과가 보여주는 것은 특정 데이터셋에서 특정하게 학습된 두 개의 단봉 프로브를 결합했을 때 성능이 개선된다는 사실이다.

다음 가능성을 배제해야 한다.

- rank-1 구조가 우연히 해당 데이터의 편향과 맞았을 가능성.
- 프로브가 서로 다른 비의미론적 단서를 학습했을 가능성.
- 정규화나 온도 파라미터의 영향.
- bilinear 헤드가 점수 크기나 임계값 조정만으로 성능을 얻었을 가능성.
- rank-2, diagonal, full-rank 헤드와의 비교가 충분하지 않은 가능성.

최소한 다음 기준선을 포함해야 한다.

- 원래 코사인 유사도.
- 랜덤 프로브의 외적.
- 학습된 scalar affine 조정.
- diagonal bilinear 헤드.
- rank-1, rank-2, rank-4, full-rank 헤드.
- 파라미터 수를 통제한 MLP 또는 선형 헤드.

이를 통해 rank-1의 구체적 구조가 중요한지, 단순히 유사도 함수를 조금이라도 학습하면 되는지를 구별할 수 있다.

## 4. 개선을 위한 실험 제안

### 4.1 인과적 개입 실험

논문의 중심 주장을 가장 직접적으로 검증하는 실험은 \(\gamma\)를 의도적으로 조절하는 것이다.

- 텍스트 임베딩에 부정 방향을 추가한다.
- 이미지와 텍스트의 극성 방향을 Procrustes 또는 whitening으로 정렬한다.
- rank-1 bilinear 헤드를 학습한다.
- \(\gamma\), \(\alpha\), \(\beta\), 2×2 정확도를 동시에 측정한다.

예상되는 결과는 다음과 같아야 한다.

\[
\gamma \uparrow,\qquad
\max(|\alpha|,|\beta|)\text{ 고정},\qquad
\Delta(S)\uparrow
\]

개입 후 실제로 \(\gamma > \max(|\alpha|,|\beta|)\)를 넘는 개념이 증가하고 검색 성능이 함께 상승한다면, “교차항이 병목이다”라는 해석이 훨씬 강해진다.

### 4.2 편집 아티팩트 통제

인페인팅 편향을 다루기 위해 다음 조건을 추가할 필요가 있다.

- 객체가 있는 이미지와 없는 이미지의 자연적 수집 데이터.
- 여러 인페인팅 모델 및 마스크 크기.
- 객체를 제거하지 않고 배경만 편집한 대조군.
- 객체를 동일 위치에 다른 객체로 교체한 대조군.
- 이미지 임베딩에서 편집 검출기를 학습하여 제거 아티팩트의 설명력을 측정하는 실험.

### 4.3 프로브의 의미론성 검증

프로브가 실제 객체 정보를 읽는지 확인하려면 다음 검사가 유용하다.

- 새로운 템플릿에 대한 전이.
- 새로운 문장 어순에 대한 전이.
- 새로운 이미지 편집 방식에 대한 전이.
- 객체 개념을 바꾼 zero-shot 평가.
- 프로브 가중치의 saliency 또는 입력 변화 분석.
- 텍스트 토큰 마스킹과 이미지 영역 마스킹 실험.

특히 텍스트의 경우 AB-swap이 단어 다중집합만 통제하고 어순은 완전히 통제하지 않으므로, 위치와 구문 구조를 모두 균형화한 템플릿이 필요하다.

### 4.4 더 강한 기준선과 외부 검증

2×2 결과만으로 일반성을 주장하기는 어렵다. 다음 평가를 추가하는 것이 좋다.

- Winoground의 group score.
- 부정 전용 벤치마크.
- 관계 및 행위 부정 데이터셋.
- 자연 이미지 기반 검색 데이터.
- 서로 다른 CLIP 계열과 학습 목적함수.
- 공개 체크포인트와 동일한 전처리 설정.

본문에 완결 논문에는 외부 벤치마크가 있다고 적혀 있으므로, 현재 판본에도 핵심 결과와 설정을 최소한 요약해야 한다. “나머지는 PAPER.md에 있다”는 설명은 제출본의 독자가 그 파일에 접근할 수 있다는 보장이 없다면 불충분하다. [ppl-ai-file-upload.s3.amazonaws](https://ppl-ai-file-upload.s3.amazonaws.com/web/direct-files/attachments/61333062/4c4ba4d0-8c5d-4fb4-8f4d-12b0f128db58/PAPER_2PAGE.pdf?AWSAccessKeyId=ASIA2F3EMEYE4KAQ7JI2&Signature=tXGsMX339QK7pQebh7nQ3VvrYrM%3D&x-amz-security-token=IQoJb3JpZ2luX2VjEM7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJGMEQCIFR25qlXD0mxPyNT01izBIY9WJYsbSL2EgHfZAk8qOxDAiBH4Jo9vl%2BoaknN%2Fxp94KuTg3%2FDj9I2Fgy8k10kXFQlsyr8BAiX%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDY5OTc1MzMwOTcwNSIM38aoc6LzpKb2TgyhKtAESW2RHQyb1TiAChs%2BIc1ECwW%2BlyLxmf0TmaAOeApr17T27P%2BxpVQgDx%2BQhEppJuU%2FrhsnzhZHaGDFLTCbZfe6C822B8NV1cpJQnRISaHc7QDbn6KruMLHRRJ7dWrbb%2FuCIttp0PMyrj4h8x7kXg10gvCDW8AXiuTD2brmhY8x3RET6E6ncie2HUxce3APk4wXY1nl6Pv9%2FuurjeLScepFYa1KU%2BAC7GzSYXFBX7NLznJVCaBbeKrUrRVNcd26gLqv1Bu5zx9gYmB2akIgfobeS%2BNDiWiSfcHXduxqK1LgNVa%2Fm%2Fpuzhww1Z3Is4Y2zE3CjUH%2FAbN5LW0QFOC8OSJj0kZWODbRfFTtwcueEp4klzDMEzhO3%2FszO%2FZR%2FmknUJvFm005kW6x%2Fb3hiEHWW8t8pbVoWTWyzxJd7gm%2BZLeE9MDCqd1Yj%2FSVmfUi0ePEoguwFk%2BNdxvt9PRuHk%2BW8eE1sPkfI5CpElCdQeMWDpvNxiFH%2Fo7%2FDzStwyD1%2BasSZpkd3MzmL1YSnmzAR%2FA9Ae1tA4q6jqlZs50AV16AyQd1rcWFdGTW4a3I2Kt96M4tFAyyJBkfpZ1FPSVP1qLWpNBCMJiLULHswa%2FlE0ZkWsnFGaa7%2BWv33z%2FUAl2TxXFtKQO8N5ziTreolMXfSqsdCyNCSuEGXuLbN%2BXihL%2Bt8GE8MqJmmrfQdNqi1my%2Bp8O%2FbRQysETEwFbhkV%2BXI%2BBV4r7IP6rcIAHwyCLsF%2B7m%2F%2BBEYGUxx8NLlPJAP9TMBvboOa4kAcLzyZetiws486l90y59iTClkNbUBjqZAQSFAuGKOpO6i0OmA79iJpdqq7%2BTpTM0pzwWkyqrAt5ybK%2FNBunE8pckT6uAJ8X29wDNdtTKJ0iTch9FbMv9wn1wd7EGCkUnVvUCmQDreUxa2cf%2Bs%2FtIJW%2F8H0q24ybivo8k7VCgBcysO6IWzXVb0oCiCDmojmjb1nbMeLCmWBa1gTZU7PA6HAx67wPhqgHMI34K3L977nXgIw%3D%3D&Expires=1788188152)

## 5. 세부 의견

### 표현과 구성

- 제목의 “CLIP의 부정 매칭 실패”는 범위가 넓다. “객체 존재 부정의 2×2 매칭 분석”처럼 조건을 제목에 반영하면 더 정확하다.
- 저자와 소속이 미정인 상태는 심사본이라도 완성도가 낮아 보일 수 있다.
- 2페이지 논문에서 수식, 통계량, 모델 비교, 외부 검증을 모두 담으려 해 결과의 정의와 실험 설정이 지나치게 압축되어 있다.
- \(\alpha,\beta,\gamma\)의 부호 규약과 상태 \(a,b\)의 의미를 표로 명시하는 것이 좋다.
- “방향이 없는 것은 주효과 쪽이다”는 표현은 다소 모호하다. “주효과의 부호가 개념별로 일관되지 않다”처럼 쓰는 편이 정확하다.
- “정보는 실재한다”는 표현은 프로브 검출 가능성을 의미하는 것으로 한정해야 한다. 그렇지 않으면 의미론적 이해나 인과적 표현을 주장하는 것처럼 읽힐 수 있다.
- \(\cos(d_I,d_T)\)는 서로 다른 모달리티의 벡터 공간 사이에서 어떻게 계산했는지 설명이 필요하다. 이미지와 텍스트 임베딩이 같은 차원의 공유 공간에 있더라도, 차이 벡터의 정렬이 의미론적으로 어떤 역할을 하는지 추가 정당화가 필요하다.
- \(r=+0.835\)의 정의가 본문에 없다. 부정 질의의 R@1 하락폭이 정확히 무엇을 뜻하는지 밝혀야 한다.
- p-value보다 효과 크기와 신뢰구간을 우선 제시하는 편이 좋다. 현재 표본 수가 크기 때문에 극도로 작은 p-value가 실질적 중요성을 대신하는 인상을 줄 수 있다.

## 최종 판정

**판정: Major Revision / Weak Reject**

이 논문은 문제의식이 분명하고, 2×2 분해를 이용해 CLIP의 부정 매칭 실패를 해석 가능한 형태로 설명하려는 시도가 인상적이다. 특히 “프로브로 검출 가능함”과 “기존 코사인 유사도에서 검색 가능함”을 구분하고, rank-1 상호작용 헤드로 이를 연결하려는 구조는 충분히 발전 가능하다.

그러나 현재 판본에서는 다음 핵심 주장이 아직 완전히 입증되지 않았다.

1. 프로브가 실제 의미론적 정보를 읽는가.
2. \(\gamma\)의 작음이 인페인팅·템플릿·데이터셋 편향의 결과가 아닌가.
3. \(\gamma\)가 단순한 기술적 분해항이 아니라 인과적 병목인가.
4. 결과가 객체 존재 부정을 넘어 일반적인 CLIP 부정 이해 실패에도 적용되는가.
5. rank-1 헤드가 특별히 필요한 구조인가.

따라서 재심사를 위해서는 데이터 분할과 통계 단위를 명확히 하고, 인페인팅 아티팩트 통제, 강한 bilinear 기준선, 개입에 따른 \(\gamma\) 변화, 외부 벤치마크 결과를 보강해야 한다. 이러한 보완이 이루어진다면, 본 논문은 CLIP의 실패를 “표현 부족”과 “상호작용 함수의 부적절성”으로 분해하는 유용한 분석 프레임워크로 발전할 가능성이 있다.

**Critical Review: “Where CLIP’s Negation Matching Fails: Represented Information versus Similarity Interaction”**

### Summary of the Contribution
The paper addresses a well-known failure mode of CLIP-style dual encoders: queries that require object absence (“a photo without a dog / pizza”). Prior diagnoses locate the problem either in the embedding geometry, in the similarity function, or claim that the necessary information is already present but unused. The authors formalize a clean 2×2 decomposition of the four matching scores into a constant, two main effects (image presence/absence \(\beta\), text affirmation/negation \(\alpha\)), and an interaction term \(\gamma\):

\[
S_{ab}=C+a\beta+b\alpha+ab\gamma,\qquad a,b\in\{+1,-1\}.
\]

They prove the identity \(\Delta(S)=2\gamma-2\max(|\alpha|,|\beta|)\), so successful ranking of the two “correct” cells over the two “incorrect” cells is exactly equivalent to \(\gamma>\max(|\alpha|,|\beta|)\). On controlled BEAF counterfactual image pairs + carefully constructed AB-swap text pairs, they measure the three coefficients and show that \(\gamma\) is systematically too small. Linear probes confirm that presence/absence information exists in both modalities, yet cosine similarity cannot exploit it. The same ordering holds across nine models. A simple rank-1 bilinear head built from the two unimodal probe normals recovers substantially higher accuracy, demonstrating that the bottleneck is the fixed identity metric rather than missing representation or high-rank structure.

### Strengths
1. **Clean formalization.** The decomposition is assumption-free (just a change of basis) and the success condition is an identity, not an approximation. This is rare and valuable; it turns a qualitative complaint into a quantitative, falsifiable claim.
2. **Controlled minimal pairs.** Using BEAF’s inpainting pairs + AB-swap texts is methodologically careful. It largely removes superficial lexical cues and forces the model to bind the negation to the correct object.
3. **Cross-model consistency.** Evaluating nine models spanning architecture, data, objective, and negation-specialized fine-tuning strengthens the claim that the phenomenon is structural rather than an artifact of one checkpoint.
4. **Bridging representation and similarity.** Measuring the same embeddings under different scoring rules (cosine vs. probe-based binary functions vs. outer-product rank-1) is an elegant way to put “information is present” and “matching fails” on a common axis.
5. **Geometric interpretation.** Relating \(\gamma\) to the cosine between the two difference vectors \(d_I\) and \(d_T\) is insightful and empirically supported (Spearman \(\rho\approx0.74\)).

### Major Weaknesses and Concerns

**1. Scope is narrower than the title and framing suggest.**  
The paper studies only object-presence negation in multi-object scenes under a highly constrained text template (AB-swap). Relation negation, action negation, quantifiers, and single-object “no-X” queries are explicitly left out. The claim that “negation matching fails because of similarity interaction” is therefore only demonstrated for one important but limited subclass. The introduction and conclusion occasionally read as if the diagnosis is general; it is not.

**2. The AB-swap design, while clever, still leaves residual cues.**  
The authors acknowledge residual order effects (accuracy drops from 73.8% to 64.7% when position is controlled). They argue the remaining signal is still well above chance, which is true, but the residual 12% relative contribution of surface order means the text probe is not purely measuring compositional binding of negation. A stricter control (e.g., fully order-matched or template-varied pairs with identical token multisets and identical positions) would have been stronger.

**3. Probe-based upper bound is optimistic and not comparable in the way claimed.**  
The 26% accuracy obtained by enumerating binary functions of the two probe bits is interesting, but it uses oracle knowledge of the concept (probes are trained per concept with 5-fold cross-validation). Cosine similarity is a fixed, concept-agnostic function. The gap therefore mixes two distinct issues: (a) the metric cannot use the information, and (b) the metric does not know which direction is the relevant polarity for this particular concept. The rank-1 outer-product head partially addresses this, but the comparison still overstates how much of the failure is purely “similarity interaction.”

**4. Statistical and numerical claims need tighter reporting.**  
- Macro numbers (\(|\alpha|=0.00380\), \(|\beta|=0.00628\), \(\gamma=0.00121\)) are given without clear aggregation method (mean of absolute values? median?).  
- “100% bit-wise agreement” between algebraic and empirical success is reassuring but expected once floating-point precision is considered; it mainly confirms implementation correctness.  
- The claim that \(\gamma\) is strictly positive for all 42 concepts (Wilcoxon \(p=2.3\times10^{-13}\)) is strong; bootstrap CIs and multiple-testing correction details would help.  
- Zero successes out of 378 concept–model pairs is striking, but without variance estimates or a null model that preserves the marginal distributions of \(\alpha,\beta,\gamma\), it is hard to judge how surprising this is.

**5. Causal interpretation of the geometric story is incomplete.**  
The authors show \(\cos(d_I,d_T)\approx0.167\) and that this correlates with \(\gamma\). This is consistent with misalignment of polarity directions, but it does not explain *why* the directions are misaligned. Is it an optimization artifact of the contrastive loss, a consequence of the bag-of-words tendency, or something about the training data distribution of negations? The paper stops at the geometric observation.

**6. Limited discussion of practical implications and remedies.**  
Showing that a rank-1 head recovers performance is useful diagnostically, but the paper does not explore whether such a head can be made concept-agnostic, whether it can be trained without concept labels, or how it interacts with retrieval-scale galleries. The external validation on COCO T2I R@1 drop (\(r=+0.835\)) is promising but thin (only the correlation is reported).

### Minor Issues
- Notation is occasionally overloaded (\(a,b\) for states, later used for other quantities).  
- Figure captions and some equation renderings in the provided PDF are hard to parse; the visual presentation of the 2×2 matrices could be clearer.  
- Related work is appropriately cited, but the positioning relative to concurrent negation papers (especially those that intervene in the embedding space) could be sharper: the decomposition actually *explains* why some of those interventions succeed or fail.

### Overall Assessment
This is a strong, focused diagnostic paper. Its main intellectual contribution—the exact algebraic characterization of when 2×2 negation ranking succeeds, together with the empirical demonstration that the interaction term is systematically insufficient—is clean and reusable. The controlled experimental design and multi-model consistency are above average for the genre.

However, the scope is narrower than the rhetoric, the upper-bound experiments mix representation quality with concept-specific polarity knowledge, and the geometric explanation stops short of a mechanistic account. As a short paper or workshop contribution it is already valuable; for a top-tier conference it would benefit from (i) stricter text controls or additional negation types, (ii) a concept-agnostic version of the bilinear recovery experiment, and (iii) a clearer statement of the limits of the claimed diagnosis.

**Recommendation:** Accept with minor-to-moderate revisions (or borderline accept / workshop if space is constrained). The core formal insight is worth disseminating even if the empirical coverage remains limited.