# 한혁진 (Hyukjin Han)
# https://orcid.org/0009-0007-4132-1707
# bokkamsun@gmail.com
# SPDX-License-Identifier: Apache-2.0
"""Atom tokenizer (banya_atoms.py)

An atom dictionary of syllables, symbols, and reserved words. Vocab about 3k. Drop-in replacement for sentencepiece.
Symbols, operators, and reserved words are whole single tokens; function and variable names are character compositions; rare Hangul syllables are decomposed into jamo.

Token groups
  Control          <pad> <unk> <nl>
  ASCII            95 printable (letters digits space single-char operators punctuation)
  Compat jamo      51 (jamo used alone, like ㅋㅋ ㅠㅠ)
  Conjoining jamo  67 (leading medial trailing jamo for decomposing rare syllables)
  Greek            math physics super/subscripts blackboard
  Operators        multi-character operators of six languages
  Keywords         C C++ assembly Python JavaScript Bash keywords
  Syllables        top 2000 most frequent in dialogue

Run  python3 core/banya_atoms.py   (self-verification)

원자 토크나이저 (banya_atoms.py)

음절 기호 예약어 원자 사전. vocab 약 3천. sentencepiece 대체(드롭인).
기호 연산자 예약어는 통짜 1토큰, 함수명 변수명은 글자 조합, 드문 한글은 자모 분해.

토큰 묶음
  제어      <pad> <unk> <nl>
  ASCII     인쇄 가능 95 (영문 숫자 공백 단문자 연산자 문장부호)
  호환자모   51 (ㅋㅋ ㅠㅠ 처럼 홀로 쓰는 자모)
  결합자모   67 (드문 음절 분해용 초성 중성 종성)
  그리스     수학 물리 첨자 블랙보드
  연산자     6개 언어 다문자 연산자
  예약어     C C++ 어셈블리 Python JavaScript Bash 키워드
  음절       대화 빈출 상위 2000

실행  python3 core/banya_atoms.py   (자가검증)
"""
import os
import unicodedata
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
SYLL_FILE = os.path.join(_HERE, "banya_syllables.txt")

# ===== Token groups =====
# ===== 토큰 묶음 =====
CONTROL = ["<pad>", "<unk>", "<nl>"]
WS = ["\t", "\r", "\x0b", "\x0c"]                                    # tab CR vertical-tab form-feed, lossless indentation for AST / 탭 CR 수직탭 폼피드. 들여쓰기 무손실(AST)
ASCII = [chr(c) for c in range(0x20, 0x7F)]                          # 95
COMPAT = [chr(c) for c in range(0x3131, 0x3164)]                     # compat jamo 51 / 호환자모 51
CONJ = ([chr(0x1100 + i) for i in range(19)]                         # conjoining leading 19 / 결합 초성 19
        + [chr(0x1161 + i) for i in range(21)]                      # conjoining medial 21 / 결합 중성 21
        + [chr(0x11A8 + i) for i in range(27)])                     # conjoining trailing 27 / 결합 종성 27
GREEK = list("αβγδεζηθικλμνξοπρστυφχψωςΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩϑϕϖϵ")
MATH = list("+−×÷±∓⋅∗⊗⊙⊘≠≈≡≢≤≥≪≫∝≅≃≜≐∼"
            "∈∉∋⊂⊃⊆⊇⊄⊅∪∩∖∅℘⋃⋂"
            "∫∬∭∮∂∇∆∑∏∐√∛∜∞′″‴∴∵"
            "→←↑↓↔↕⇒⇐⇑⇓⇔↦↪⟶⟵⟹⟺"
            "⌈⌉⌊⌋⟨⟩⟦⟧‖∥∠∟·…⋯⋮⋱"
            "∧∨¬⊕⊻⊼⊽⊤⊥⊢⊨∀∃∄")
SUPSUB = list("⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿⁱ₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₓᵢⱼₖₙ")
BBB = list("ℝℂℤℕℚℙℍ𝔽")
PHYS = list("°℃℉ÅΩµℏħ∿‰‱")
# Extension for dynamics, geometry, and linear algebra. Specified by code point because there are many combining marks and rare symbols
# 동역학 기하 선형대수 확장. 결합표식과 희귀 기호가 많아 코드포인트로 지정
EXTMATH = [chr(c) for c in [
    0x0307, 0x0308, 0x20D7, 0x0302, 0x0305, 0x0303, 0x030C,   # combining: dot double-dot vector-arrow hat overline tilde check above / 결합: 점 두점 벡터화살 모자 윗줄 물결 체크(위)
    0x0300, 0x0301, 0x0304, 0x0306, 0x030A, 0x0327, 0x0323,   # combining accents: grave acute macron breve ring cedilla underdot, precise European decomposition / 결합 악센트: 그레이브 아큐트 마크롱 브레베 링 세디유 아래점(유럽어 정밀분해)
    0x2113, 0x2112, 0x210B, 0x2110, 0x211B, 0x2130, 0x2131,   # script: ℓ ℒ ℋ ℐ ℛ ℰ ℱ, dynamics Lagrangian Hamiltonian / 스크립트: ℓ ℒ ℋ ℐ ℛ ℰ ℱ (동역학 라그랑지안 해밀토니안)
    0x2020, 0x2021,                                           # † ‡ Hermitian conjugate and footnote / † ‡ 에르미트 켤레 각주
    0x25B3, 0x25BD, 0x25A1, 0x25CB, 0x25C7, 0x2221, 0x2222,   # geometric shapes △ ▽ □ ○ ◇ ∡ ∢ / 기하 도형 △ ▽ □ ○ ◇ ∡ ∢
    0x2312, 0x224C, 0x2237, 0x22BE,                           # ⌒ arc ≌ congruence ∷ proportion ⊾ right angle / ⌒ 호 ≌ 합동 ∷ 비례 ⊾ 직각
    0x1D40, 0x2218, 0x22C6,                                   # linear algebra ᵀ transpose ∘ composition Hadamard ⋆ star operator / 선형대수 ᵀ 전치 ∘ 합성아다마르 ⋆ 별연산
    0x220E, 0x2254, 0x225D, 0x225F, 0x2057,                   # proof and definition ∎ ≔ ≝ ≟ ⁗ / 증명 정의 ∎ ≔ ≝ ≟ ⁗
    0x222F, 0x2230, 0x1D53C, 0x2201, 0x2223, 0x2224, 0x2AEB,  # probability and calculus ∯ ∰ 𝔼 expectation ∁ complement ∣ conditional ∤ does-not-divide ⫫ independence / 확률 미적분 ∯ ∰ 𝔼기댓값 ∁여집합 ∣조건부 ∤나눔없음 ⫫독립
    0x1D9C, 0x1D48, 0x1D4F, 0x1D57, 0x1D40, 0x207F,           # superscript lowercase ᶜ complement ᵈ ᵏ ᵗ ᵀ duplicate ⁿ / 위첨자 소문자 ᶜ여집합 ᵈ ᵏ ᵗ ᵀ(중복) ⁿ
]]

OPS = ["<<=", ">>=", ">>>=", "->*", "<=>", "->", ".*", "::", "++", "--",
       "<<", ">>", "<=", ">=", "==", "!=", "&&", "||", "+=", "-=", "*=",
       "/=", "%=", "&=", "|=", "^=", "...", "**", "**=", "//", "//=", ":=",
       "=>", "===", "!==", "??", "?.", "??=", "&&=", "||=", ">>>", "${",
       "#!", ";;", "&>", ">&", "2>", "|&", "$(", "<<<", "/*", "*/"]

KW_C = ("auto break case char const continue default do double else enum extern "
        "float for goto if inline int long register restrict return short signed "
        "sizeof static struct switch typedef union unsigned void volatile while "
        "_Bool _Complex _Atomic").split()
KW_CPP = ("alignas alignof and and_eq bitand bitor bool catch class compl concept "
          "constexpr constinit consteval const_cast decltype delete dynamic_cast "
          "explicit export false friend mutable namespace new noexcept not not_eq "
          "nullptr operator or or_eq private protected public reinterpret_cast "
          "requires static_assert static_cast template this thread_local throw true "
          "try typeid typename using virtual wchar_t xor xor_eq co_await co_return "
          "co_yield").split()
KW_PY = ("and as assert async await break class continue def del elif else except "
         "finally for from global if import in is lambda nonlocal not or pass raise "
         "return try while with yield None True False match case self").split()
KW_JS = ("var let const function return if else for while do switch case break "
         "continue new delete typeof instanceof void this super class extends static "
         "get set import export default from as async await yield try catch finally "
         "throw of in null undefined true false").split()
KW_BASH = ("if then elif else fi for while until do done case esac function in select "
           "time coproc echo cd export local return source alias unset read test").split()
KW_ASM = ("mov lea push pop add sub mul imul div idiv inc dec cmp test jmp je jne jg "
          "jl call ret nop int syscall and or xor not shl shr rax rbx rcx rdx rsi rdi "
          "rbp rsp eax ebx r8 r9 xmm0").split()
KEYWORDS = KW_C + KW_CPP + KW_PY + KW_JS + KW_BASH + KW_ASM


def load_syllables():
    with open(SYLL_FILE, encoding="utf-8") as f:
        return list(f.read())


def build_itos():
    syll = load_syllables()
    groups = [CONTROL, WS, ASCII, COMPAT, CONJ, GREEK, MATH, SUPSUB, BBB, PHYS,
              EXTMATH, OPS, KEYWORDS, syll]
    seen = set()
    itos = []
    for g in groups:
        for tok in g:
            if tok not in seen:
                seen.add(tok)
                itos.append(tok)
    return itos


def _is_id_char(c):
    return c.isascii() and (c.isalnum() or c == "_")


class AtomTokenizer:
    """Atom-dictionary tokenizer. Drop-in for sentencepiece (vocab itos stoi encode decode).

    원자 사전 토크나이저. sentencepiece 드롭인(vocab itos stoi encode decode)."""

    def __init__(self, text=None, model_file=None):
        self.itos = build_itos()
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.vocab = len(self.itos)
        self.model_file = None
        self.pad = self.stoi["<pad>"]
        self.unk = self.stoi["<unk>"]
        self.nl = self.stoi["<nl>"]
        self.kw = set(KEYWORDS)
        self.ops = sorted(set(OPS), key=len, reverse=True)     # match longer operators first / 긴 연산자 먼저 매칭
        self.opfirst = set(op[0] for op in self.ops)           # prunes operator match attempts / 연산자 시도 가지치기
        singles = [(t, i) for i, t in enumerate(self.itos) if len(t) == 1]   # lookup table for single-character tokens / 단일글자 토큰 룩업표
        maxcp = max(ord(t) for t, _ in singles)
        self._lut = np.full(maxcp + 1, -1, dtype=np.int64)
        for t, i in singles:
            self._lut[ord(t)] = i
        self._lut[0x0A] = self.nl                              # newline maps to <nl> / 개행은 <nl> 로
        self._opcps = np.array(sorted(set(ord(ch) for op in OPS for ch in op)), dtype=np.uint32)

    def encode(self, text):
        # Vectorized path. Single characters are handled in bulk via the LUT; only special positions (keywords, multi-character operators, rare syllables, unregistered chars)
        # go through scalar logic. The result is exactly identical to _encode_scalar
        # 벡터화 경로. 단일글자는 LUT 로 일괄, 특수 위치(예약어 다문자연산자 드문음절 미등록)만
        # 스칼라 로직으로 처리한다. 결과는 _encode_scalar 와 완전히 동일
        if not text:
            return np.zeros(0, dtype=np.int64)
        cps = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
        n = len(cps)
        base = np.full(n, -1, dtype=np.int64)
        inr = cps < self._lut.shape[0]
        base[inr] = self._lut[cps[inr]]
        idc = (((cps >= 97) & (cps <= 122)) | ((cps >= 65) & (cps <= 90))
               | ((cps >= 48) & (cps <= 57)) | (cps == 95))    # identifier characters a-z A-Z 0-9 _ / 식별자 글자 a-z A-Z 0-9 _
        opc = np.isin(cps, self._opcps)                        # characters used in operators / 연산자에 쓰이는 글자

        def adj(mask):                                         # whether the same class adjoins on either side / 좌우로 같은 부류가 붙어있나
            a = np.zeros(n, dtype=bool)
            a[1:] |= mask[:-1]
            a[:-1] |= mask[1:]
            return a
        special = (base < 0) | (idc & adj(idc)) | (opc & adj(opc))
        if not special.any():
            return base                                        # fast path for spans of pure syllables and whitespace / 순수 음절 공백 구간 빠른 경로
        flips = np.nonzero(np.diff(special.view(np.int8)))[0] + 1   # boundaries between special and plain spans / 특수 평범 구간 경계
        bounds = np.concatenate(([0], flips, [n]))
        parts = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            if special[a]:
                parts.append(self._encode_scalar(text[a:b]))   # keywords operators and syllable decomposition go through the scalar path / 예약어 연산자 음절분해는 스칼라로
            else:
                parts.append(base[a:b])
        return np.concatenate(parts)

    def _encode_scalar(self, text):
        out = []
        stoi = self.stoi
        i = 0
        n = len(text)
        while i < n:
            c = text[i]
            if c == "\n":
                out.append(self.nl)
                i += 1
                continue
            if c.isascii() and (c.isalpha() or c == "_"):      # identifier run: whole token if keyword, else per character / 식별자 런: 예약어면 통짜, 아니면 글자
                j = i + 1
                while j < n and _is_id_char(text[j]):
                    j += 1
                word = text[i:j]
                if word in self.kw:
                    out.append(stoi[word])
                else:
                    for ch in word:
                        out.append(stoi.get(ch, self.unk))
                i = j
                continue
            if c in self.opfirst:                              # longest match for multi-character operators / 다문자 연산자 최장일치
                op = None
                for cand in self.ops:
                    if text.startswith(cand, i):
                        op = cand
                        break
                if op is not None:
                    out.append(stoi[op])
                    i += len(op)
                    continue
            if c in stoi:                                      # single registered character / 단일 등록 글자
                out.append(stoi[c])
                i += 1
                continue
            oc = ord(c)
            if 0xAC00 <= oc <= 0xD7A3:                         # rare syllable decomposed into jamo / 드문 음절 자모 분해
                code = oc - 0xAC00
                cho = code // 588
                jung = (code % 588) // 28
                jong = code % 28
                out.append(stoi[chr(0x1100 + cho)])
                out.append(stoi[chr(0x1161 + jung)])
                if jong:
                    out.append(stoi[chr(0x11A8 + jong - 1)])
                i += 1
                continue
            dec = unicodedata.normalize("NFD", c)              # unregistered char: use the precise decomposition if every piece is registered / 미등록 글자: 정밀분해가 전부 등록조각이면 그걸로
            if len(dec) > 1 and all(d in stoi for d in dec):   # Ĥ -> H + hat, é -> e + acute; composed and decomposed forms converge / Ĥ -> H + 모자, é -> e + 아큐트. 합쳐진 형태와 분해 형태 수렴
                for d in dec:
                    out.append(stoi[d])
            else:
                out.append(self.unk)
            i += 1
        return np.array(out, dtype=np.int64)

    def _compose(self, s):
        # Refolds runs of conjoining jamo into syllables. Only the leading medial (trailing) order is folded; everything else passes through unchanged
        # 결합자모 연속을 음절로 되접는다. 초성 중성 (종성) 순서만 접고 나머지는 그대로
        res = []
        i = 0
        n = len(s)
        while i < n:
            c = s[i]
            oc = ord(c)
            if 0x1100 <= oc <= 0x1112 and i + 1 < n and 0x1161 <= ord(s[i + 1]) <= 0x1175:
                cho = oc - 0x1100
                jung = ord(s[i + 1]) - 0x1161
                jong = 0
                if i + 2 < n and 0x11A8 <= ord(s[i + 2]) <= 0x11C2:
                    jong = ord(s[i + 2]) - 0x11A8 + 1
                    i += 3
                else:
                    i += 2
                res.append(chr(0xAC00 + (cho * 21 + jung) * 28 + jong))
            else:
                res.append(c)
                i += 1
        return "".join(res)

    def decode(self, ids):
        buf = []
        for t in ids:
            t = int(t)
            if t == self.nl:
                buf.append("\n")
            elif t == self.pad:
                continue
            elif t == self.unk:
                buf.append("�")
            else:
                buf.append(self.itos[t])
        return self._compose("".join(buf))


if __name__ == "__main__":
    tok = AtomTokenizer()
    itos = tok.itos
    print(f"vocab 총 {tok.vocab}")
    # Counts per group (actual contribution after deduplication)
    # 묶음별 개수(중복 제거 후 실제 반영분)
    syll = load_syllables()
    for name, g in [("제어", CONTROL), ("ASCII", ASCII), ("호환자모", COMPAT),
                    ("결합자모", CONJ), ("그리스", GREEK), ("수학", MATH),
                    ("첨자", SUPSUB), ("블랙보드", BBB), ("물리", PHYS),
                    ("연산자", sorted(set(OPS))), ("예약어", sorted(set(KEYWORDS))),
                    ("음절", syll)]:
        uniq = len(set(g))
        print(f"  {name:8s} 목록 {len(g):5d}  고유 {uniq:5d}")
    print(f"  예약어 6개 언어 합 {len(KEYWORDS)} -> 중복제거 {len(set(KEYWORDS))}")

    print("\n=== 왕복 검증 ===")
    tests = {
        "C코드": "int x = a==b ? 1 : 0;  // 주석\nreturn x->y;",
        "파이썬": "def f(a, b):\n    return a**2 + b // 3  # 나눗셈",
        "JS": "const g = (x) => x ?? 0;  let y = a===b;",
        "Bash": "if [ $x -gt 3 ]; then echo hi >&2; fi",
        "수식": "∑ xᵢ ≤ ∫ f(x)dx,  α+β=γ,  x∈ℝ,  ∂u/∂t",
        "한글": "안녕하세요 반야입니다 ㅋㅋㅋ ㅠㅠ 진짜?",
        "혼합": "변수 result = tensor·W + b;  θ를 학습",
    }
    allok = True
    for name, s in tests.items():
        ids = tok.encode(s)
        back = tok.decode(ids)
        ok = (back == s)
        allok &= ok
        print(f"  [{name}] 토큰 {len(ids):3d}  왕복 {'OK' if ok else '깨짐'}")
        if not ok:
            print(f"     원본: {s!r}")
            print(f"     복원: {back!r}")

    # Verifies the forced decomposition path for rare syllables
    # 드문 음절 강제 분해 경로 검증
    rare = [chr(c) for c in range(0xAC00, 0xD7A4) if chr(c) not in tok.stoi]
    print(f"\n사전 밖(자모 분해 대상) 음절 {len(rare):,}개")
    import random
    random.seed(0)
    sample = "".join(random.sample(rare, 200))
    ids = tok.encode(sample)
    back = tok.decode(ids)
    print(f"  드문 음절 200개 왕복 {'OK' if back == sample else '깨짐'}  (평균 {len(ids)/200:.1f} 토큰/자)")
    allok &= (back == sample)

    print(f"\n전체 {'전부 통과' if allok else '실패 있음'}")
