// Project page has no interactive components beyond native <details>.
// Kept as a hook for future additions (e.g. carousels).
document.addEventListener('DOMContentLoaded', function () {
  // Open the "More examples" block automatically when the page is printed.
  window.addEventListener('beforeprint', function () {
    document.querySelectorAll('details').forEach(function (d) { d.open = true; });
  });
});
