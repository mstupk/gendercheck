def replace_foo_with_bar():
    import uno
    from com.sun.star.beans import PropertyValue
    from com.sun.star.text.ControlCharacter import PARAGRAPH_BREAK

    desktop = XSCRIPTCONTEXT.getDesktop()
    model = desktop.getCurrentComponent()

    if not model or not model.supportsService("com.sun.star.text.TextDocument"):
        return

    text = model.Text
    cursor = text.createTextCursor()

    while True:
        found = text.findFirst("foo", cursor.End, False)
        if not found:
            break
        found.setString("bar")
