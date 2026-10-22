import { useEffect } from 'react';

export interface ToastMessage {
  id: number; // a new message restarts the timer even when its words repeat
  text: string;
}

const SHOW_MS = 8000;

// The short message an action leaves (the server's own words: "Daily predictor: queued"), at the foot of the screen
// over everything, read out by a screen reader; it goes after a few seconds or on its ×. The URL never carries it.
export default function Toast({ toast, onClose }: { toast: ToastMessage | null; onClose: () => void }) {
  useEffect(() => {
    if (!toast) return undefined;
    const timer = setTimeout(onClose, SHOW_MS);
    return () => clearTimeout(timer);
    // a new message restarts the timer
  }, [toast?.id]);
  return (
    <div className="toast-slot" role="status" aria-live="polite">
      {toast ? (
        <div className="toast">
          <span>{toast.text}</span>
          <button type="button" onClick={onClose} aria-label="Dismiss the message">
            ×
          </button>
        </div>
      ) : null}
    </div>
  );
}
