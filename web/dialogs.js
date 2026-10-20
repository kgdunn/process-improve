// web/dialogs.js: what the app's dialogs do besides what the browser gives them. Escape closes a modal dialog
// already; a click on the backdrop, outside the dialog's box, closes it too.

/** Whether a pointer event lands outside an element's box. */
function outside(element, event) {
  const box = element.getBoundingClientRect();
  return event.clientX < box.left || event.clientX > box.right || event.clientY < box.top || event.clientY > box.bottom;
}

/**
 * Close ``dialog`` on a click on its backdrop. The press and the release must both land outside its box: text
 * selected inside the dialog, with the button let go outside it, leaves the dialog open, so the text can be copied.
 * A click on the backdrop is the dialog's own, as the backdrop belongs to it.
 */
export function closeOnBackdrop(dialog) {
  let pressedOutside = false;
  dialog.addEventListener("pointerdown", (event) => {
    pressedOutside = event.target === dialog && outside(dialog, event);
  });
  dialog.addEventListener("click", (event) => {
    const close = pressedOutside && event.target === dialog && outside(dialog, event);
    pressedOutside = false;
    if (close) dialog.close();
  });
}
