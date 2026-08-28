import unohelper
import uno
from com.sun.star.task import XJobExecutor

class Foo2BarJob(unohelper.Base, XJobExecutor):

    def __init__(self, ctx):
        self.ctx = ctx
        sm = ctx.getServiceManager()
        self.desktop = sm.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)

    def trigger(self, args):
        doc = self.desktop.getCurrentComponent()
        if not hasattr(doc, "Text"):
            return
        text = doc.Text
        cursor = text.createTextCursor()
        cursor.gotoStart(False)
        while True:
            found = doc.findFirst("foo", cursor.End, False)
            if not found:
                break
            found.String = "bar"
            cursor = text.createTextCursorByRange(found.End)

# Register the component
g_ImplementationHelper = unohelper.ImplementationHelper()
g_ImplementationHelper.addImplementation(
    Foo2BarJob,
    "org.example.foo2bar.Foo2BarJob",
    ("com.sun.star.task.Job",),
)
