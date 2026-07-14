import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

type MarkdownAnswerProps = {
  content: string;
};

export function MarkdownAnswer({ content }: MarkdownAnswerProps) {
  return (
    <div className="markdown-answer">
      <ReactMarkdown
        components={{
          a: ({ children, href }) => (
            <a href={href} rel="noreferrer" target="_blank">
              {children}
            </a>
          ),
          table: ({ children }) => (
            <div className="markdown-table-wrap">
              <table>{children}</table>
            </div>
          ),
        }}
        remarkPlugins={[remarkGfm]}
        skipHtml
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
