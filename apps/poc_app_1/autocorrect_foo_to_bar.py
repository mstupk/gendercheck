# autocorrect_foo_to_bar.py
# Modified to work as a LibreOffice Python macro
import uno
from com.sun.star.awt import XKeyListener
from com.sun.star.lang import XServiceInfo
from com.sun.star.uno import XInterface
from com.sun.star.uno import Any
from com.sun.star.uno import Type
from com.sun.star.connection import NoConnectException

class FooBarReplacer(XKeyListener, XServiceInfo):
    """Implements XKeyListener to handle key events and replace 'foo' with 'bar'."""
    def __init__(self, ctx, controller):
        self.ctx = ctx
        self.controller = controller
        self.buffer = ""

    # XKeyListener methods
    def keyPressed(self, key_event):
        """Handle key press events to detect 'foo' and replace with 'bar'."""
        # Check for printable characters
        if key_event.KeyChar and key_event.KeyChar >= 32:
            char = chr(key_event.KeyChar)
            self.buffer += char

            # Check if buffer ends with 'foo' (case-sensitive)
            if self.buffer.endswith("foo"):
                doc = self.controller.getModel()
                text = doc.Text
                cursor = self.controller.getViewCursor()

                # Move cursor back 3 characters to select 'foo'
                cursor.goLeft(3, True)
                cursor.setString("bar")  # Replace 'foo' with 'bar'
                self.buffer = self.buffer[:-3] + "bar"  # Update buffer

        # Reset buffer on space or enter
        elif key_event.KeyCode in (1280, 1281):  # Key codes for Space (1280) and Enter (1281)
            self.buffer = ""

        return False  # Allow event to propagate

    def keyReleased(self, key_event):
        """Handle key release events (no action needed)."""
        return False

    def disposing(self, event):
        """Handle disposal of the listener."""
        pass

    # XServiceInfo methods (required for UNO component)
    def getImplementationName(self):
        return "FooBarReplacer"

    def supportsService(self, service_name):
        return service_name == "com.sun.star.awt.XKeyListener"

    def getSupportedServiceNames(self):
        return ("com.sun.star.awt.XKeyListener",)

def register_handler():
    """Register the key handler with the current LibreOffice document."""
    # Get component context
    ctx = uno.getComponentContext()
    smgr = ctx.ServiceManager

    # Get the desktop and current document
    try:
        desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
        doc = desktop.getCurrentComponent()
        if not doc:
            raise Exception("No active document found")
        controller = doc.getCurrentController()
        if not controller:
            raise Exception("No active controller found")

        # Create and register the key handler
        handler = FooBarReplacer(ctx, controller)
        window = controller.getFrame().getContainerWindow()
        window.addKeyListener(handler)

        # Store handler to prevent garbage collection
        global _handlers
        _handlers.append(handler)
        return True
    except Exception as e:
        print(f"Error registering handler: {e}")
        return False

# Global list to keep handlers alive
_handlers = []

# Trigger registration when script runs
if __name__ == "__main__":
    success = register_handler()
    if success:
        print("FooBarReplacer handler registered successfully")
    else:
        print("Failed to register FooBarReplacer handler")
