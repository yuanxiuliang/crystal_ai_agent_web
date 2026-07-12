import type { Citation, RetrievalTrace, RouteDecision } from "../../lib/types";

type EvidencePanelProps = {
  citations: Citation[];
  route: RouteDecision | null;
  retrieval: RetrievalTrace | null;
  currentNode: string | null;
  activity: string[];
};

export function EvidencePanel({ citations, route, retrieval, currentNode, activity }: EvidencePanelProps) {
  return (
    <aside className="evidence-panel">
      <section className="panel-section">
        <h2 className="section-title">Activity</h2>
        {currentNode ? <div className="trace-box">正在：{currentNode}</div> : <p className="muted">空闲</p>}
        {activity.length > 0 ? (
          <div className="trace-box">
            {activity.map((item, index) => (
              <div key={`${item}-${index}`}>{item}</div>
            ))}
          </div>
        ) : null}
      </section>

      <section className="panel-section">
        <h2 className="section-title">Route</h2>
        {route ? (
          <div className="trace-box">
            <div>intent: {route.intent}</div>
            <div>retrieve: {route.should_retrieve ? "yes" : "no"}</div>
            <div>{route.reason}</div>
          </div>
        ) : (
          <p className="muted">暂无路由决策</p>
        )}
      </section>

      <section className="panel-section">
        <h2 className="section-title">Retrieval</h2>
        {retrieval ? (
          <div className="trace-box">
            <div>query: {retrieval.query}</div>
            <div>top_k: {retrieval.top_k}</div>
            <div>results: {retrieval.result_count}</div>
            <div>sufficient: {String(retrieval.sufficient)}</div>
          </div>
        ) : (
          <p className="muted">本轮尚无检索</p>
        )}
      </section>

      <section className="panel-section">
        <h2 className="section-title">Citations</h2>
        {citations.length === 0 ? (
          <p className="muted">暂无引用</p>
        ) : (
          citations.map((citation) => (
            <article className="citation-card" key={citation.record_id}>
              <div className="record-id">{citation.record_id}</div>
              <div className="muted">score: {citation.score.toFixed(2)}</div>
              {citation.doi ? <div className="muted">doi: {citation.doi}</div> : null}
              <div className="source-text">{citation.source_text}</div>
            </article>
          ))
        )}
      </section>
    </aside>
  );
}
