"""
Gendercheck grammar-checker UNO component.

Implements com.sun.star.linguistic2.XProofreader so LibreOffice's native
grammar-checking pipeline finds gendering suggestions the same way it finds
any other grammar issue: a colored underline plus a right-click "Correct"
menu with the suggested replacement. This is the §6.1-decided suggest (not
auto-apply) UX, using LibreOffice's own suggestion UI rather than custom
overlay code -- see SPECIFICATION.md §13 for why this supersedes the
XKeyListener design originally sketched in §4.

Talks to inference_server.py over local HTTP (SPECIFICATION.md §4.2: don't
load a transformer model inside LibreOffice's bundled Python interpreter).
If the inference server isn't reachable, doProofreading degrades to
returning no errors rather than raising -- a missing suggestion is a much
smaller problem for the user than a broken grammar checker.
"""

import json
import urllib.request
import urllib.error

import unohelper
from com.sun.star.linguistic2 import XProofreader
from com.sun.star.lang import XServiceInfo, XInitialization
from com.sun.star.linguistic2 import ProofreadingResult
from com.sun.star.linguistic2 import SingleProofreadingError
from com.sun.star.beans import PropertyValue
from com.sun.star.text.TextMarkupType import PROOFREADING as MARKUP_PROOFREADING

IMPLEMENTATION_NAME = "org.gendercheck.Proofreader"
SUPPORTED_SERVICE_NAMES = ("com.sun.star.linguistic2.Proofreader",)
INFERENCE_SERVER_URL = "http://127.0.0.1:8765/check"
REQUEST_TIMEOUT_SECONDS = 2.0


def _query_inference_server(text):
    """Return a list of candidate dicts from the local inference server, or
    [] on any failure (connection refused, timeout, malformed response)."""
    try:
        body = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(
            INFERENCE_SERVER_URL, data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode("utf-8")).get("candidates", [])
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        return []


class GendercheckProofreader(unohelper.Base, XProofreader, XServiceInfo, XInitialization):
    def __init__(self, ctx, *args):
        # LibreOffice's real internal linguistic framework instantiates
        # registered checkers via createInstanceWithArgumentsAndContext,
        # which passes extra (currently unused) constructor arguments even
        # when none are configured -- a plain __init__(self, ctx) raises
        # TypeError on that call path and LO silently drops the checker
        # from its active set, while a manual test call via the simpler
        # createInstanceWithContext (single ctx arg) succeeds regardless,
        # masking the problem. Accept and ignore *args for exactly this
        # reason. (Confirmed against a real-world report of the same
        # failure mode: https://keithcu.com/wordpress/?p=5276)
        self.ctx = ctx

    # ---- XServiceInfo ----
    def getImplementationName(self):
        return IMPLEMENTATION_NAME

    def supportsService(self, service_name):
        return service_name in SUPPORTED_SERVICE_NAMES

    def getSupportedServiceNames(self):
        return SUPPORTED_SERVICE_NAMES

    # ---- XInitialization ----
    def initialize(self, args):
        pass

    # ---- XProofreader ----
    def isSpellChecker(self):
        return False

    def getLocales(self):
        import uno
        loc = uno.createUnoStruct("com.sun.star.lang.Locale")
        loc.Language = "de"
        loc.Country = "DE"
        loc.Variant = ""
        return (loc,)

    def hasLocale(self, locale):
        return locale.Language == "de"

    def ignoreRule(self, rule_id, locale):
        pass

    def resetIgnoreRules(self):
        pass

    def doProofreading(self, doc_id, text, locale, start_pos, suggested_end_pos, properties):
        result = ProofreadingResult()
        result.aDocumentIdentifier = doc_id
        result.aText = text
        result.aLocale = locale
        result.nStartOfSentencePosition = start_pos
        result.nBehindEndOfSentencePosition = suggested_end_pos if suggested_end_pos > 0 else len(text)
        result.xProofreader = self
        result.aProperties = ()

        segment = text[start_pos:result.nBehindEndOfSentencePosition]
        candidates = _query_inference_server(segment)

        errors = []
        for c in candidates:
            err = SingleProofreadingError()
            err.nErrorStart = start_pos + c["start"]
            err.nErrorLength = c["end"] - c["start"]
            err.nErrorType = MARKUP_PROOFREADING
            err.aRuleIdentifier = "org.gendercheck.suggestion"
            err.aSuggestions = (c["suggestion"],)
            err.aShortComment = "Gendering suggestion"
            err.aFullComment = (
                f"'{c['word']}' could be gendered as '{c['suggestion']}' "
                f"(confidence {c['confidence']:.0%})."
            )
            err.aProperties = ()
            errors.append(err)

        result.aErrors = tuple(errors)
        return result


def createInstance(ctx, *args):
    # See the matching comment on GendercheckProofreader.__init__: the real
    # UNO factory-function call site (pyuno's ImplementationHelper wiring
    # into createInstanceWithArgumentsAndContext) can pass extra args too.
    return GendercheckProofreader(ctx, *args)


g_ImplementationHelper = unohelper.ImplementationHelper()
g_ImplementationHelper.addImplementation(
    createInstance, IMPLEMENTATION_NAME, SUPPORTED_SERVICE_NAMES,
)
