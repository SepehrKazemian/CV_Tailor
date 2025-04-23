from nltk.stem import PorterStemmer, WordNetLemmatizer

# Initialize once
_ps = PorterStemmer()
_lm = WordNetLemmatizer()

def stem_token(token: str) -> str:
    """Return the Porter‑stemmed form of a single token."""
    return _ps.stem(token)

def lemmatize_token(token: str, pos: str = 'n') -> str:
    """
    Return the lemmatized form of a single token.
    pos: 'n'=noun, 'v'=verb, 'a'=adj, 'r'=adv – default to noun.
    """
    return _lm.lemmatize(token, pos=pos)

def _normalize(tok: str, type : str = "lemmatize") -> str:
    """
    Lowercase + lemmatize as a noun.
    Falls back to the original token if lemmatizer returns the same.
    """
    t = tok.lower()
    if type == "lemmatize":
        return lemmatize_token(t, pos="n")
    if type == "stem":
        return stem_token(t)

def apply_stems_or_lemmas(phrases: list[str], _stemming: bool = True) -> list[str]:
    """
    For each phrase, tokenize and either stem or lemmatize every token,
    then rejoin. E.g. ["Frameworks"] → ["framework", ...].
    """
    out = []
    for ph in phrases:
        toks = ph.split()
        new = " ".join(lemmatize_token(t.lower()) for t in toks)
        out.append(new)
        if _stemming:
            new = " ".join(stem_token(t.lower()) for t in toks)
            out.append(new)
    return list(dict.fromkeys(out))