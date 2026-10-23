export function LoadingBlock({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="empty" role="status">
      <span className="spinner" aria-hidden="true" /> {label}
    </div>
  )
}
