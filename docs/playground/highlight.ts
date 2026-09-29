/** Minimal TypeScript syntax highlighting for the generated snippet (DOM-safe: no innerHTML). */

const TOKEN =
    /(\/\/[^\n]*)|("(?:[^"\\\n]|\\.)*")|\b(import|from|const|new|declare|export|true|false)\b|\b(\d+(?:\.\d+)?)\b/g;

export function highlightTypeScript(target: HTMLElement, code: string): void {
    const nodes: Node[] = [];
    let last = 0;
    for (const match of code.matchAll(TOKEN)) {
        const index = match.index ?? 0;
        if (index > last) nodes.push(document.createTextNode(code.slice(last, index)));
        const span = document.createElement("span");
        span.className = match[1] ? "tok-com" : match[2] ? "tok-str" : match[3] ? "tok-kw" : "tok-num";
        span.textContent = match[0];
        nodes.push(span);
        last = index + match[0].length;
    }
    if (last < code.length) nodes.push(document.createTextNode(code.slice(last)));
    target.replaceChildren(...nodes);
}
