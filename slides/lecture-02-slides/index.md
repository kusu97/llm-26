<section class="title">
  <div class="title-main">Lecture 02 - 𝑵-gram Language Models</div>
  <div class="title-sub">NLP and LLMs (CS40008.01)</div>
  <div class="title-meta">
    <div>Baojian Zhou</div>
    <div>School of Data Science</div>
    <div>Fudan University</div>
    <div>03/05/2026</div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Outline</div>
  <div class="ppt-line"></div>
  <ul class="outline-bullets big">
  <li class="active">Probabilistic N-gram LMs</li>
  <li class="muted">Evaluating LMs and Perplexity</li>
  <li class="muted">Smoothing N-gram LMs</li>
  <li class="muted">Neural Proabablistic LMs</li>
  </ul>
</section>

---

<section class="ppt">
  <div class="ppt-title">Assign probabilities to sentences</div>
  <div class="ppt-line"></div>
  <ul style="font-size: 50px; line-height: 1.1; font-weight: 600; margin-top: 1px;">
      <ul style="font-size: 42px; margin-top: 20px;">
        <li class="fragment">Speech Recognition
          <ul style="font-size: 32px; margin-top: 20px; margin-bottom: 20px; font-weight: 500;">
            <li>$P($<span class="text-green">It's hard to recognize speech</span>$) > P($<span class="text-red">It's hard to wreck a nice beach</span>$)$</li>
          </ul>
        </li>
        <li class="fragment">Machine Translation (MT)
          <ul class="spaced-list" style="font-size: 32px; margin-top: 20px; margin-bottom: 20px; font-weight: 500;">
            <li>"他向记者介绍了主要内容" is translated into 4 candidates:</li>
            <li>$S_1$ = <span class="text-green">He briefed reporters on the main contents of the statement</span></li>
            <li>$S_2$ = He introduced reporters to the main contents of the statement</li>
            <li>$S_3$ = He briefed to reporters the main contents of the statement</li>
            <li>$S_4$ = <span class="text-red">He to reporters introduced main content</span></li>
            <li>A good MT model should have $P($<span class="text-green">$S_1$</span> $)> P(S_2) \approx P(S_3) >P($ <span class="text-red">$S_4$</span>$)$</li>
          </ul>
        </li>
        <li class="fragment">Spell Correction
          <ul style="font-size: 32px; margin-top: 20px; margin-bottom: 20px; font-weight: 500;">
            <li>The office is about fifteen <span class="text-red">minuets</span> from my house</li>
            <li>$P($<span class="text-green">about fifteen minutes from</span>$) > P($<span class="text-red">about fifteen minuets from</span>$)$</li>
          </ul>
        </li>
      </ul>
  </ul>
</section>

---

<!-- Slide 1: The setting -->
<section class="ppt">
  <div class="ppt-title">LLMs approximate an unknown distribution</div>
  <div class="ppt-line"></div>

  <div style="font-size: 34px; line-height: 1.25; font-weight: 550; margin-top: 18px;">
    <div class="fragment" data-fragment-index="1">
      <span style="font-weight: 700;">Real text-based data (languages, code, etc)</span> induces a distribution over token sequences:
      $$\mathbf{w}_{1:n} = (w_1,\dots,w_n), \quad \mathbf{w}_{1:n} \sim p_{\text{data}}(\cdot),$$
      where <span class="text-red" style="font-weight: 700;">$p_{\text{data}}$ is unknown</span>. We only observe a corpus (samples):
      $$\mathcal{D}=\{\mathbf{w}^{(i)}\}_{i=1}^N,\quad \mathbf{w}^{(i)}\overset{\text{i.i.d.}}{\sim} p_{\text{data}}(\cdot).$$
    </div>
    <div class="fragment" data-fragment-index="3" style="margin-top: 14px;">
      We build and train a model distribution (LLMs) $p_\theta(\mathbf{w})$ such that
      $$p_\theta(\mathbf{w}) \approx p_{\text{data}}(\mathbf{w}).$$
    </div>
    <div class="fragment" data-fragment-index="4" style="margin-top: 18px; padding: 14px 18px; border-radius: 14px; background: rgba(0,0,0,0.06);">
      <span style="font-weight: 800;">In $p_\theta$,</span>
      sequences that look like real language should get
      <span class="text-green" style="font-weight: 800;">higher</span> probability than corrupted/random ones:
      $p_\theta(\text{“natural sentence”})\;>\;p_\theta(\text{“random sentence”}).$
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Training objective: make \(p_\theta \approx p_{\text{data}}\)</div>
  <div class="ppt-line"></div>
  <div style="font-size: 34px; line-height: 1.25; font-weight: 550; margin-top: 18px;">
    <div class="fragment" data-fragment-index="1">
      Ideal goal (distribution matching) is to make $p_\theta \approx p_{\text{data}}.$ To do this, one principled way is to find a $\theta$ such that the KL divergence is minimized, i.e.,
      $$\theta^\star=\arg\min_\theta \mathrm{KL}\big(p_{\text{data}} \,\|\, p_\theta\big).$$
    </div>
    <div class="fragment" data-fragment-index="3" style="margin-top: 5px;">
      This is equivalent to maximizing expected log-likelihood:
      $$\arg\min_\theta \mathrm{KL}(p_{\text{data}}\|p_\theta)
      \;\Longleftrightarrow\;
      \arg\max_\theta \mathbb{E}_{\mathbf{w} \sim p_{\text{data}}}\big[\log p_\theta(\mathbf{w})\big].$$
    </div>
    <div class="fragment" data-fragment-index="4" style="margin-top: 5px;">
      Since \(p_{\text{data}}\) is unknown, use the dataset $\{\mathbf{w}^{(i)}\}_{i=1}^N$ to approximate the expectation:
      $$\max_\theta \frac{1}{N}\sum_{i=1}^N \log p_\theta (\mathbf{w}^{(i)}).$$
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Training objective: make \(p_\theta \approx p_{\text{data}}\)</div>
  <div class="ppt-line"></div>
  <div style="font-size: 34px; line-height: 1.25; font-weight: 550; margin-top: 18px;">
    <div class="fragment" data-fragment-index="4" style="margin-top: 5px;">
      Since \(p_{\text{data}}\) is unknown, use the dataset $\{\mathbf{w}^{(i)}\}_{i=1}^N$ to approximate the expectation:
      $$\max_\theta \frac{1}{N}\sum_{i=1}^N \log p_\theta (\mathbf{w}^{(i)}).$$
    </div>
    <div class="fragment" data-fragment-index="5" style="margin-top: 5px;">
      Let $\mathbf{w}^{(i)} := (w_1^{(i)},\ldots,{w}_{n_i}^{(i)})$. The LM factorization (what the model actually learns):
      $$p_\theta(\mathbf{w}^{(i)})=\prod_{t=1}^{n_i} p_\theta \left(w_t^{(i)} \mid w_{1:t-1}^{(i)} \right).$$
      <div style="margin-top: 8px; opacity: 0.95;">
        So training teaches the model to predict the next token like real text →
        sampling from \(p_\theta\) produces human-like sequences.
      </div>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Training samples from real-world</div>
  <div class="ppt-line"></div>

  <div style="display: flex; gap: 28px; align-items: flex-start; margin-top: 22px;">
    <!-- Left: dataset notation -->
    <div style="font-size: 40px; font-weight: 650; line-height: 1.2;">
      $$\mathcal{D}=\left\{\mathbf{w}^{(i)}\right\}_{i=1}^{N}=$$
    </div>
    <div style="
  background: rgba(0,0,0,0.06);
  border: 1px solid rgba(0,0,0,0.14);
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.05);
  max-width: 1100px;
">
  <div style="display: flex; gap: 14px; align-items: flex-start;">
    <div style="font-size: 34px; line-height: 1.25; font-weight: 560;">
      <div style="margin-bottom: 12px;">\(\mathbf{w}^{(1)}\) : It′s hard to recognize speech.</div>
      <div style="margin-bottom: 12px;">\(\mathbf{w}^{(2)}\) : He briefed reporters on the main contents of the statement.</div>
      <div style="margin-bottom: 12px;">\(\mathbf{w}^{(3)}\) : The office is about fifteen minutes from my house.</div>
      <div style="margin-bottom: 12px;">\(\mathbf{w}^{(4)}\) : I want to learn how to play the guitar.</div>
      <div style="opacity: 0.85; margin: 6px 0;">\(\vdots\)</div>
      <div style="opacity: 0.9;">$\mathbf{w}^{(i)}$ : #include<stdio.h> int main(void){ printf (“Hello, world!\n”);}</div>
      <div style="opacity: 0.85; margin: 8px 0 0 0;">\(\vdots\)</div>
    </div>
  </div>
</div>
  </div>

  <div class="fragment" style="margin-top: 22px; font-size: 34px; font-weight: 560;">
    Assume $\mathbf{w}^{(i)} \overset{\text{i.i.d.}}{\sim} p_{\text{data}}$ and train $p_\theta$ to fit these samples.
  </div>
</section>

---

<!-- (Optional) small CSS helpers (put once near the top of your reveal HTML) -->
<style>
  .ppt-title { font-size: 52px; font-weight: 900; color: #0b2d6b; }
  .ppt-line  { height: 6px; background: #0b2d6b; margin: 10px 0 18px 0; }
  .two-col { display: flex; gap: 40px; align-items: flex-start; }
  .col { flex: 1; }
  .bigul { font-size: 28px; line-height: 1.22; }
  .bigul > li { margin: 12px 0; }
  .subul { font-size: 24px; line-height: 1.25; margin-top: 10px; }
  .subul > li { margin: 8px 0; }
  .text-red { color: #d62728; font-weight: 800; }
  .text-green { color: #2ca02c; font-weight: 800; }
  .text-blue { color: #1f77b4; font-weight: 800; }
  .mathline { margin: 10px 0 0 0; }
</style>

<!-- Slide 1 -->
<section class="ppt">
  <div class="ppt-title">Compute \(P(w_{1:t})\)</div>
  <div class="ppt-line"></div>

  <div class="two-col">
    <!-- Left column -->
    <div class="col">
      <ul class="bigul">
        <li>
          Write sentence \(\mathbf{w}=[w_1,w_2,\ldots, w_n]\) as \(w_{1:n}\),
          <div class="mathline">
            \[
              P(\mathbf{w}) \equiv P(w_{1:n})
            \]
          </div>
        </li>
        <li>
          Let $X_1 = w_1, X_2 = w_2$, recall the Chain rule
          \[ P(X_1X_2)=P(X_1)\cdot P(X_2\mid X_1) \]
        </li>
        <li>
          Applying the Chain rule to sentence \(w_{1:n}\)
          <div class="mathline">
            \[
              P(w_{1:n})=\prod_{t=1}^{n} P\!\left(w_t \mid w_{1:t-1}\right)
            \]
          </div>
          <ul class="subul">
            <li> \(w_{1:t-1}\) is called <span class="text-green">history</span> of \(w_t\)</li>
            <li>We assume \(w_{1:0}=w_0=\text{BOS}\).</li>
            <li>We assume last token \(w_n=\text{EOS}\).</li>
            <li>
              When applying the Chain rule, to compute \(P(w_{1:n})\) is essentially a
              <span class="text-green">next word prediction problem</span>
            </li>
          </ul>
        </li>
      </ul>
    </div>
    <div class="col">
      <div style="height:100px;"></div>
      <div class="fragment"  style="
        background: rgba(0,0,0,0.06);
        border-radius: 14px;
        padding: 14px 16px;
        font-size: 30px;
        line-height: 1.25;
        font-weight: 650;
      ">
        <span class="text-red">Key success of almost all modern LLMs: Better next-token prediction ⇒ better language modeling.</span><br/>
      </div>
      <div style="height:20px;"></div>
      <div class="fragment" style="margin-top: 16px;">
        <!-- Replace this with your video tag / embed -->
        <div style="
          width: 100%;
          height: 260px;
          border-radius: 14px;
          background: rgba(0,0,0,0.12);
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 22px;
          opacity: 0.9;
        ">
        <video controls style="width:100%; height:260px; border-radius:14px;"><source src="media/predicting-next-word.mp4" type="video/mp4"></video>
        </div>
      </div>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">The \(N\)-gram model</div>
  <div class="ppt-line"></div>

  <ul class="bigul">
    <li>
      <b>Intuition:</b> instead of using entire history \(w_{1:t-1}\), we can
      <span class="text-red">approximate</span> probability by using the last few \((N)\) words
    </li>
    <li><div style="height:20px;"></div>
      <b>Unigram model</b> \((N=1)\)
      <ul class="subul">
        <li>Approximates \(P(\cdot)\) without history: $\quad P(w_t \mid w_{1:t-1}) \approx q(w_t)$</li>
        <li> Example: $P(\text{skills}\mid \text{I want to improve my cooking}) \approx P(\text{skills})$ </li>
      </ul>
    </li>
    <li><div style="height:20px;"></div>
      <b>Bigram model</b> \((N=2)\)
      <ul class="subul">
        <li>Approximates \(P(\cdot)\) by only using \(w_{t-1}\): $\quad P(w_t \mid w_{1:t-1}) \approx q(w_i \mid w_{t-1})$</li>
        <li>Example: $P(\text{skills}\mid \text{I want to improve my cooking}) \approx q(\text{skills}\mid \text{cooking})$</li>
      </ul>
    </li>
    <li><div style="height:20px;"></div>
      <b>Trigram model</b> \((N=3)\)
      <ul class="subul">
        <li>Approximates \(P(w_t\mid w_{1:t-1})\) by only using \(w_{t-2:t-1}\): $\quad P(w_i\mid w_{1:t-1}) \approx q(w_i \mid w_{t-2:t-1})$</li>
        <li>Example: $P(\text{skills}\mid \text{I want to improve my cooking}) \approx q(\text{skills}\mid \text{my cooking})$</li>
      </ul>
    </li>
  </ul>

  <div  class="fragment" style="font-size: 34px; line-height: 1.25; font-weight: 550; margin-top: 18px;">
    <ul style="font-size: 30px; line-height: 1.25;">
      <li>
        <b>The above approximations use the <b>Markov assumption</b>. In general, for \(N\)-gram</b> \((N\ge 2)\):
      </li>
      \[
          \text{(N-1)-order Markov:}\qquad P(w_t \mid w_{1:t-1}) \approx q(w_t \mid w_{t-N+1:t-1}).
        \]
    </ul>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Build N-gram LMs</div>
  <div class="ppt-line"></div>
  <div class="two-col">
    <!-- Left: key definitions -->
    <div class="col">
      <ul class="bigul" style="font-size: 32px;">
        <li><b>Step 1</b>: Pick up an order $N$</li>
        <li><b>Step 2</b>: Curate your training data \(\mathcal{D}\) and vocabulary $\mathcal{V}$
          <ul style="font-size: 0.85em; margin-top: 10px;">
          <div style="height:10px;"></div>
          <li>Step 2.1: Collect raw data (tokenization, filtering)</li>
          <div style="height:10px;"></div>
          <li>Step 2.2: Build vocabulary $\mathcal{V}$ after tokenization </li>
          </ul>
        </li>
        <li><b>Step 3</b>: Estimate your paramters $$P(w_t\mid w_{t-N+1:t-1})$$ (<b>Q: How many parameters will we have ?</b>)</li>
        <li><b>Step 4</b>: Test your model by generating sentences or estimating log probabilities (or perplexity) of given sentence in test dataset</li>
      </ul>
    </div>
    <div class="col">
      <div style="
        background: rgba(0,0,0,0.06);
        border-radius: 14px;
        padding: 14px 16px;
        font-size: 20px;
        line-height: 1.25;
        font-weight: 650;
        text-align: center;
      "><span class="text-red">Whiteboard: </span> Estimate bigram $P(w_t|w_{t-1})$
      </div>
      <div style="height:30px;"></div>
      <div class="fragment", style="font-size: 20px;">
      Let $\mathcal{V}=\{v_1,v_2,\ldots,v_{|V|}\}$.
      Define all possible paramters \[
        \theta_{i,j} = P(w_t=v_i|w_{t-1}=v_j)
        \]
      \[\boldsymbol{\theta}
      = \begin{bmatrix}
      \theta_{11} & \theta_{12} & \theta_{13} & \cdots & \theta_{1 |\mathcal{V}|}\\
      \theta_{21} & \theta_{22} & \theta_{23} & \cdots & \theta_{2 |\mathcal{V}|}\\
      \theta_{31} & \theta_{32} & \theta_{33} & \cdots & \theta_{3 |\mathcal{V}|}\\
      \vdots      & \vdots      & \vdots      & \ddots & \vdots     \\
      \theta_{|\mathcal{V}| 1} & \theta_{|\mathcal{V}| 2} & \theta_{|\mathcal{V}| 3} & \cdots & \theta_{|\mathcal{V}| |\mathcal{V}|}
      \end{bmatrix}
      \]
      </div>
      <div style="height:10px;"></div>
      <div class="fragment", style="font-size: 20px;">
        <div style="height:10px;"></div>
        <li><b>MLE for bigram model:</b>
        $$p_\theta(v_i|v_j) = \frac{C(v_i v_j)}{C(v_i)},$$ where $C(v_i v_j)$ counts total frequency of bigram $[v_i,v_j]$ in training corpus. </li>
        <div style="height:10px;"></div>
        <li><b>Potential parameters ($N=2$)</b>: $\mathcal{O}(|\mathcal{V}|^2)$</li>
        <div style="height:10px;"></div>
        <li><b>$N$-gram is not scalable: $\mathcal{O}(|\mathcal{V}|^N)$</b></li>
      </div>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Toy example of training bigram LM</div>
  <div class="ppt-line"></div>

  <div style="font-size: 25px; line-height: 1.22; font-weight: 560; margin-top: 14px;">
    <ul style="margin-top: 6px;">
      <li class="fragment" data-fragment-index="1">
        <b>Maximum likelihood estimate</b> (bigram model):
        \[
          p_\theta(w_i \mid w_{i-1})
          = \frac{C(w_{i-1}w_i)}{\sum_{w\in \mathcal{V}} C(w_{i-1}w)}
          = \frac{C(w_{i-1}w_i)}{C(w_{i-1})}.
        \]
        <ul style="font-size: 24px; margin-top: 8px; line-height: 1.25;">
          <li>\(C(x)\): frequency of \(x\) in the corpus</li>
          <li>\(\mathcal{V}\): vocabulary of running text</li>
        </ul>
      </li>
      <li class="fragment" data-fragment-index="2" style="margin-top: 10px;">
        <b>An example of training corpus ($\mathcal{V}=\{EOS, I, am, Sam, do, not, like, eggs, and, ham\}$)</b> as follows
        <div style="
          margin-top: 10px;
          background: rgba(0,0,0,0.06);
          border: 1px solid rgba(0,0,0,0.12);
          border-radius: 14px;
          padding: 12px 14px;
          font-size: 20px;
          line-height: 1.35;
          font-weight: 650;
        ">
          <div><span class="text-blue">BOS</span> <span class="text-green">I am Sam</span> <span class="text-blue">EOS</span></div>
          <div style="margin-top: 6px;"><span class="text-blue">BOS</span> <span class="text-green">Sam I am</span> <span class="text-blue">EOS</span></div>
          <div style="margin-top: 6px;"><span class="text-blue">BOS</span> <span class="text-green">I do not like eggs and ham</span> <span class="text-blue">EOS</span></div>
        </div>
      </li>
      <li class="fragment" data-fragment-index="3" style="margin-top: 10px;">
        <b>Then we have part of parameter estimations</b>
        <div style="display: flex; gap: 28px; margin-top: 10px; font-size: 28px; line-height: 1.25;">
          <div style="flex: 1;">
            \[
              p_\theta(I\mid \text{BOS})=\frac{2}{3}=0.67
            \]
            \[
              p_\theta(\text{EOS}\mid Sam)=\frac{1}{2}=0.50
            \]
          </div>
          <div style="flex: 1;">
            \[
              p_\theta(Sam\mid \text{BOS})=\frac{1}{3}=0.33
            \]
            \[
              p_\theta(Sam\mid am)=\frac{1}{2}=0.50
            \]
          </div>
          <div style="flex: 1;">
            \[
              p_\theta(am\mid I)=\frac{2}{3}=0.67
            \]
            \[
              p_\theta(do\mid I)=\frac{1}{3}=0.33
            \]
          </div>
        </div>
      </li>
    </ul>

  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Bigram statistics from restaurant reviews</div>
  <div class="ppt-line"></div>

  <div class="two-col">
    <div class="col">
      <ul class="bigul" style="font-size: 32px;">
        <li>
          \(\mathcal{D}\): restaurant reviews (e.g., 1000 sentences)
        </li>
        <li style="margin-top: 10px;">
          Example sentences from reviews
          <ul class="subul" style="font-size: 26px;">
            <li>Wow... Loved this place ...</li>
            <li>Not tasty and the texture was just nasty ...</li>
            <li>The selection on the menu was great and ...</li>
          </ul>
        </li>
        <li style="margin-top: 10px;">
          Add boundary tokens: <span class="text-blue">BOS</span> ... <span class="text-blue">EOS</span>
        </li>
      </ul>
    <div style="height:50px;"></div>
    <div style="
        background: rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.12);
        border-radius: 14px;
        padding: 14px 16px;
        font-size: 28px;
        line-height: 1.25;
        font-weight: 650;
      ">
        <div style="margin-bottom: 8px;">Bigram counts: $C(w_{i-1},w_i)$ </div>
        <div style="margin-bottom: 8px;">Bigram MLE: $p_\theta(w_i\mid w_{i-1})=\frac{C(w_{i-1},w_i)}{C(w_{i-1})}$</div>
        <div style="opacity: 0.95; font-weight: 560;">
          We can visualize a small sub-matrix for selected words
          (e.g., <span class="text-green">i, want, to, eat, in, this, place</span>).</div>
      </div>
    </div>
    <div class="col">
    <div style="height:100px;"></div>
    <div style="display:flex; gap:4px; align-items:flex-start;">
        <pre style="
          background: rgba(0,0,0,0.04);
          margin: 0;
          font-size: 20px;
          line-height: 1.25;
          white-space: pre;
        ">
      Bigram counts (subset)<br></br>
      split: train rows: 3000
      columns: ['text']
      Vocab size: 5166
      Distinct bigrams: 22179<br></br>
              i   want   to  eat  in  this  place
      i       0     5     0   1   0    0     0
      want    0     0    11   0   0    0     0
      to      0     1     0  12   2    6     2
      eat     0     0     0   0   2    0     0
      in      0     0     3   0   0   24     2
      this    1     0     4   0   2    0    73
      place   2     0    10   0   3    0     0
      <br></br></pre>
      </div>
    </div>
  </div>
</section>

---

<!-- One concise reveal slide summarizing the 3 images -->
<section class="ppt">
  <div class="ppt-title">Practical issues in \(N\)-gram LMs</div>
  <div class="ppt-line"></div>

  <div class="two-col">
    <!-- Left: sentence boundaries -->
    <div class="col">
      <div style="
        background: rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.12);
        border-radius: 14px;
        padding: 14px 16px;
        font-size: 30px;
        line-height: 1.25;
        font-weight: 600;
      ">
        <div style="font-weight: 800; margin-bottom: 10px;">
          1) Sentence boundaries
        </div>
        <div>
          Need extra context at the start/end.
        </div>
        <div style="margin-top: 10px;">
          For trigram:
          \[
            q(w_1\mid \text{BOS},\text{BOS})
          \]
        </div>
        <div style="margin-top: 6px; opacity: 0.95;">
          Add tokens: <span class="text-blue">BOS</span> … <span class="text-blue">EOS</span>
        </div>
      </div>
      The word like, Thisisahardtofindword simply did not occur in our training set but could be in our test test
    </div>
    <!-- Right: unknown words (OOV) -->
    <div class="col">
      <div style="
        background: rgba(0,0,0,0.06);
        border: 1px solid rgba(0,0,0,0.12);
        border-radius: 14px;
        padding: 14px 16px;
        font-size: 30px;
        line-height: 1.25;
        font-weight: 600;
      ">
        <div style="font-weight: 800; margin-bottom: 10px;">
          2) Unknown words (OOV)
        </div>
        <ul style="margin: 0; padding-left: 26px; font-size: 28px; line-height: 1.25;">
          <li class="fragment">
            <b>Closed vocabulary:</b> test words must be in a fixed lexicon
          </li>
          <li class="fragment" style="margin-top: 8px;">
            <b>Open vocabulary:</b> map unseen words to a pseudo-token
            <span class="text-blue">&lt;UNK&gt;</span>
          </li>
          <li class="fragment" style="margin-top: 10px;">
            <b>How to train</b> \(P(\text{UNK}\mid \cdot)\)?
            <ul style="margin-top: 6px; font-size: 26px; line-height: 1.22;">
              <li class="fragment">
                <b>Prior vocab:</b> convert OOV in training to <span class="text-blue">&lt;UNK&gt;</span>, then count it
              </li>
              <li class="fragment">
                <b>No prior vocab:</b> replace rare words (freq &lt; \(n\)) with <span class="text-blue">&lt;UNK&gt;</span>
                (or keep top \(|V|\) words)
              </li>
            </ul>
          </li>
        </ul>
      </div>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Outline</div>
  <div class="ppt-line"></div>
  <ul class="outline-bullets big">
  <li class="muted">Probabilistic N-gram LMs</li>
  <li class="active">Evaluating LMs and Perplexity</li>
  <li class="muted">Smoothing N-gram LMs</li>
  <li class="muted">Neural Proabablistic LMs</li>
  </ul>
</section>

---

<section class="ppt">
  <div class="ppt-title">Building LMs and evaluation</div>
  <div class="ppt-line"></div>

  <div style="display:flex; gap:26px; align-items:stretch; margin-top: 8px;">
    <!-- LEFT COLUMN: upper = split bar, lower = extrinsic -->
    <div style="flex: 1; display:flex; flex-direction:column; gap:18px;">
      <!-- Upper-left: Train/Val/Test bar -->
      <div style="
        font-size: 34px; 
        font-weight: 900;
        height: 60px;
      ">Split your data into:</div>
      <div style="
        border-radius: 14px;
        overflow: hidden;
        display: flex;
        border: 1px solid rgba(0,0,0,0.15);
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        height: 110px;
      ">
        <div style="
          flex: 7;
          background: #f2f2f2;
          display:flex; align-items:center; justify-content:center;
          font-size: 40px; font-weight: 900;
        ">
          Training Data
        </div>
        <div style="
          flex: 2.9;
          background: #e6f2dc;
          display:flex; align-items:center; justify-content:center;
          font-size: 34px; font-weight: 900; text-align:center;
        ">
          Validation<br/>Data
        </div>
        <div style="
          flex: 2.9;
          background: #dfe9f7;
          display:flex; align-items:center; justify-content:center;
          font-size: 34px; font-weight: 900; text-align:center;
        ">
          Testing<br/>Data
        </div>
      </div>
      <!-- Lower-left: Extrinsic evaluation -->
      <div style="
        flex: 1;
        background: rgba(0,0,0,0.05);
        border: 1px solid rgba(0,0,0,0.12);
        border-radius: 14px;
        padding: 16px 18px;
      ">
        <div style="font-size: 34px; font-weight: 900; margin-bottom: 10px;">
          Extrinsic evaluation
        </div>
        <ul style="font-size: 26px; line-height: 1.25; margin: 0; padding-left: 26px;">
          <li>Compare models via <b>downstream tasks</b></li>
          <li style="margin-top: 8px;">
            Examples:
            <ul style="margin-top: 6px; font-size: 24px; line-height: 1.22;">
              <li>Spell correction accuracy</li>
              <li>Machine translation accuracy</li>
            </ul>
          </li>
          <li style="margin-top: 8px;">
            <span class="text-red"><b>Time-consuming</b></span> (can take days/weeks)
          </li>
        </ul>
      </div>
    </div>
    <!-- RIGHT COLUMN: Intrinsic evaluation -->
    <div class="fragment" style="
      flex: 1;
      background: rgba(0,0,0,0.05);
      border: 1px solid rgba(0,0,0,0.12);
      border-radius: 14px;
      padding: 16px 18px;
    ">
      <div style="font-size: 34px; font-weight: 900; margin-bottom: 10px;">
        Intrinsic evaluation
      </div>
      <ul style="font-size: 26px; line-height: 1.25; margin: 0; padding-left: 26px;">
        <li>Does the LM prefer <b>good</b> sentences to <b>bad</b> ones?</li>
        <li style="margin-top: 8px;">Train on <b>training</b>, evaluate on unseen <b>test</b></li>
        <li style="margin-top: 8px;">Never leak test sentences into training</li>
        <li style="margin-top: 8px;">
          Probability-based metric:<br></br>
          <b>higher log-likelihood on test ⇒ better LM</b>
        </li>
      </ul>
      <div style="
        margin-top: 14px;
        background: rgba(0,0,0,0.06);
        border-radius: 12px;
        padding: 12px 14px;
        font-size: 24px;
        line-height: 1.25;
        font-weight: 650;
      ">
      A reasonable metric:<br></br>
        \[
          \frac{1}{|\mathcal{D}_{\text{test}}|}
          \sum_{\mathbf{w}\in\mathcal{D}_{\text{test}}}
          \log p_\theta(\mathbf{w})
        \]
      </div><div style="height:50px;"></div>
    <div style="
        background: rgba(0,0,0,0.06);
        border-radius: 14px;
        padding: 14px 16px;
        font-size: 20px;
        line-height: 1.25;
        font-weight: 650;
        text-align: center;
      "><span class="text-red">Whiteboard: </span> Estimate bigram $P(w_t|w_{t-1})$
      </div>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Perplexity</div>
  <div class="ppt-line"></div>

  <div style="display:flex; gap:26px; align-items:stretch; margin-top: 10px;">
    <!-- LEFT: Definition -->
    <div style="
      flex: 1.1;
      background: rgba(0,0,0,0.05);
      border: 1px solid rgba(0,0,0,0.12);
      border-radius: 14px;
      padding: 16px 18px;
    ">
      <div style="font-size: 34px; font-weight: 900; margin-bottom: 10px;">
        Definition (intrinsic metric)
      </div>
      <ul style="font-size: 26px; line-height: 1.25; margin: 0; padding-left: 26px;">
        <li>
          A better LM assigns <b>higher probability</b> to an unseen test set
          \(\;s_{1:m}\).
        </li>
        <li style="margin-top: 10px;">
          <b>Perplexity</b> is the inverse probability of the test set,
          normalized by length \(\,T=\sum_i |s_i|\):
        </li>
      </ul>
      <div style="
        margin-top: 12px;
        background: rgba(255,255,255,0.65);
        border: 1px solid rgba(0,0,0,0.10);
        border-radius: 12px;
        padding: 12px 14px;
        font-size: 26px;
        font-weight: 650;
      ">
        \[
          \mathrm{PPL}(s_{1:m})
          = P(s_{1:m})^{-1/T}
          = \exp\!\left(-\frac{1}{T}\log P(s_{1:m})\right)
        \]
      </div>
      <div style="margin-top: 10px; font-size: 24px; opacity: 0.95;">
        Minimizing PPL \(\Longleftrightarrow\) maximizing test probability.
      </div>
    </div>
    <!-- RIGHT: Interpretation as branching factor -->
    <div class="fragment" style="
      flex: 0.9;
      background: rgba(0,0,0,0.05);
      border: 1px solid rgba(0,0,0,0.12);
      border-radius: 14px;
      padding: 16px 18px;
    ">
      <div style="font-size: 34px; font-weight: 900; margin-bottom: 10px;">
        Interpretation
      </div>
      <ul style="font-size: 26px; line-height: 1.25; margin: 0; padding-left: 26px;">
        <li>
          Perplexity \(\approx\) <b>effective branching factor</b>:
          how many plausible next tokens the model considers.
        </li>
        <li style="margin-top: 10px;">
          Example: random digits \(\{0,\dots,9\}\), uniform guess
          \(P=1/10\) for each digit.
        </li>
      </ul>
      <div style="
        margin-top: 12px;
        background: rgba(255,255,255,0.65);
        border: 1px solid rgba(0,0,0,0.10);
        border-radius: 12px;
        padding: 12px 14px;
        font-size: 26px;
        font-weight: 650;
      ">
        \[
          \mathrm{PPL}(s)=P(w_{1:t})^{-1/t}
          =\left(\left(\tfrac{1}{10}\right)^t\right)^{-1/t}
          =10
        \]
        </div>
        <ul style="font-size: 26px; line-height: 1.25; margin: 0; padding-left: 26px;">
         <li style="margin-top: 10px;">If the model is totally random, it needs \(|V|\) guesses on average to get the next word right.</li>
         <li style="margin-top: 10px;">If $|\mathcal{V}|=1$, it surely picks right one. Hence, $\mathrm{PPL}(s_{1:m})=1$</li></div>
      </ul>

  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Lower perplexity – better model</div>
  <div class="ppt-line"></div>

  <div style="font-size: 32px; line-height: 1.22; font-weight: 560; margin-top: 12px;">
    <ul style="margin-top: 0;">
      <li class="fragment" data-fragment-index="1">
        Training 38 million words, test 1.5 million words (WSJ)
      </li>
    </ul>
    <!-- Table -->
    <div class="fragment" data-fragment-index="2" style="margin: 18px 0 18px 52px;">
      <table style="
        border-collapse: collapse;
        font-size: 28px;
        min-width: 820px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        border-radius: 10px;
        overflow: hidden;
      ">
        <thead>
          <tr style="background: #3d67c7; color: white;">
            <th style="padding: 10px 16px; text-align:center; font-weight: 900;">N-gram Order</th>
            <th style="padding: 10px 16px; text-align:center; font-weight: 900;">Unigram</th>
            <th style="padding: 10px 16px; text-align:center; font-weight: 900;">Bigram</th>
            <th style="padding: 10px 16px; text-align:center; font-weight: 900;">Trigram</th>
          </tr>
        </thead>
        <tbody>
          <tr style="background: rgba(61,103,199,0.14);">
            <td style="padding: 10px 16px; text-align:center; font-weight: 800;">Perplexity</td>
            <td style="padding: 10px 16px; text-align:center; font-weight: 800;">962</td>
            <td style="padding: 10px 16px; text-align:center; font-weight: 800;">170</td>
            <td style="padding: 10px 16px; text-align:center; font-weight: 800;">109</td>
          </tr>
        </tbody>
      </table>
    </div>
    <ul style="margin-top: 0;">
      <li class="fragment" data-fragment-index="3">
        The improvement in perplexity does <b>not</b> guarantee an (extrinsic) improvement in
        downstream tasks like speech recognition or MT.
      </li>
      <li class="fragment" data-fragment-index="4" style="margin-top: 10px;">
        Because perplexity often correlates with such improvements, it is commonly used as a
        <b>quick check</b> on an algorithm.
      </li>
      <li class="fragment" data-fragment-index="5" style="margin-top: 10px;">
          <a href="https://arxiv.org/pdf/2005.14165.pdf"
            target="_blank"
            rel="noopener noreferrer"
            style="color:#1f6feb; font-weight:800; text-decoration: underline;">
            https://arxiv.org/pdf/2005.14165.pdf
          </a>
          <span style="opacity: 0.92;">(See how GPT-3 uses PPL)</span>
      </li>
    </ul>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Sentence sampling</div>
  <div class="ppt-line"></div>

  <div style="display:flex; gap:26px; align-items:stretch; margin-top: 10px;">
    <!-- LEFT: bullets + image at bottom-left -->
    <div style="flex: 1; display:flex; flex-direction:column; justify-content:space-between;">
      <!-- bullets -->
      <div>
        <ul class="bigul" style="font-size: 32px; line-height: 1.22;">
          <li style="margin-bottom: 10px;">
            <b>Unigram case</b>
            <ul class="subul" style="font-size: 26px; line-height: 1.25; margin-top: 10px;">
              <li>
                Partition \([0,1]\) into intervals, each word gets an interval proportional to its frequency.
              </li>
              <li>
                Sample \(u\sim \mathrm{Unif}[0,1]\), output the word whose interval contains \(u\).
              </li>
              <li>
                Repeat until we generate <span class="text-green">EOS</span>.
              </li>
            </ul>
          </li>
        </ul>
      </div>
      <!-- bottom-left image -->
      <div style="margin-top: 5px;">
        <img
          data-src="media/sentence-sampling.png"
          alt="Unigram sampling on [0,1]"
          style="width: 92%; display:block; border-radius: 2px; border: 1px solid rgba(0,0,0,0.12);"
        />
        <li class="fragment"  data-fragment-index="1" style="margin-top: 14px; font-size: 30px;">
            <b>What about bigram case?</b>
          </li>
      </div>
    </div>
    <!-- RIGHT: sample outputs -->
    <div style="
      flex: 1.05;
      background: rgba(0,0,0,0.05);
      border: 1px solid rgba(0,0,0,0.12);
      border-radius: 14px;
      padding: 16px 18px;
    ">
      <div class="fragment" data-fragment-index="2" style="font-size: 34px; font-weight: 900; margin-bottom: 10px;">
        Samples from built LM (trained by the Wall Street Journal (40M))
      </div>
      <div style="
        background: rgba(255,255,255,0.70);
        border: 1px solid rgba(0,0,0,0.10);
        border-radius: 12px;
        padding: 12px 14px;
        font-size: 24px;
        line-height: 1.28;
        margin-bottom: 12px;
      " class="fragment" data-fragment-index="3">
        <b>1-gram:</b>
        Months the my and issue of year foreign new exchange’s September
        were recession exchange new endorsed a acquire to six
        <span class="text-blue"><b>executives</b></span>
      </div>
      <div style="
        background: rgba(255,255,255,0.70);
        border: 1px solid rgba(0,0,0,0.10);
        border-radius: 12px;
        padding: 12px 14px;
        font-size: 24px;
        line-height: 1.28;
        margin-bottom: 12px;
      " class="fragment" data-fragment-index="3">
        <b>2-gram:</b>
        Last December through the way to preserve the
        <span class="text-blue"><b>Hudson corporation</b></span>
        ... would seem to complete the major central planners ...
        <span class="text-blue"><b>M. X. corporation</b></span>
        ...
      </div>
      <div style="
        background: rgba(255,255,255,0.70);
        border: 1px solid rgba(0,0,0,0.10);
        border-radius: 12px;
        padding: 12px 14px;
        font-size: 24px;
        line-height: 1.28;
      " class="fragment" data-fragment-index="3">
        <b>3-gram:</b>
        They also point to ninety nine point
        <span class="text-blue"><b>six billion dollars</b></span>
        ... 
        <span class="text-blue"><b>six three percent</b></span>
        ...
        <span class="text-blue"><b>on market conditions</b></span>
      </div>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Outline</div>
  <div class="ppt-line"></div>
  <ul class="outline-bullets big">
  <li class="muted">Probabilistic N-gram LMs</li>
  <li class="muted">Evaluating LMs and Perplexity</li>
  <li class="active">Smoothing N-gram LMs</li>
  <li class="muted">Neural Proabablistic LMs</li>
  </ul>
</section>

---


<section class="ppt">
  <div class="ppt-title">Motivation: zero probabilities</div>
  <div class="ppt-line"></div>
  <div style="font-size: 30px; line-height: 1.22; font-weight: 560; margin-top: 14px;">
    <ul style="margin-top: 0; padding-left: 28px;">
      <li data-fragment-index="1">
        <b>Maximum Likelihood Estimate (trigram):</b>
        \[
          q(w_i \mid w_{i-2}w_{i-1})
          = \frac{C(w_{i-2}w_{i-1}w_i)}{C(w_{i-2}w_{i-1})}
        \]
      </li>
      <li data-fragment-index="2" style="margin-top: 8px;">
        Given the following datasets
      </li>
    </ul>
    <div class="fragment" data-fragment-index="3"
     style="display:flex; gap:32px; margin: 10px 0 8px 0; align-items: stretch;">
  <!-- Training set -->
  <div style="
    flex: 1;
    background: rgba(0,0,0,0.05);
    border: 1px solid rgba(0,0,0,0.12);
    border-radius: 14px;
    padding: 14px 16px;
  ">
    <div style="font-size: 34px; font-weight: 900; margin-bottom: 10px;">
      <span class="text-green">Training set</span>
    </div>
    <div style="font-size: 30px; line-height: 1.25; font-weight: 650;">
      <div style="margin: 8px 0; color:#3d67c7;">... denied the allegations</div>
      <div style="margin: 8px 0; color:#3d67c7;">... denied the reports</div>
      <div style="margin: 8px 0; color:#3d67c7;">... denied the claims</div>
      <div style="margin: 8px 0; color:#3d67c7;">... denied the request</div>
    </div>
  </div>
  <!-- Test set -->
  <div style="
    flex: 1;
    background: rgba(0,0,0,0.05);
    border: 1px solid rgba(0,0,0,0.12);
    border-radius: 14px;
    padding: 14px 16px;
  ">
    <div style="font-size: 34px; font-weight: 900; margin-bottom: 10px;">
      <span class="text-green">Test set</span>
    </div>
    <div style="font-size: 30px; line-height: 1.25; font-weight: 650;">
      <div style="margin: 8px 0; color:#d62728;">... denied the offer</div>
      <div style="margin: 8px 0; color:#d62728;">... denied the loan</div>
      <div style="margin-top: 14px; font-size: 32px; font-weight: 900; color:#d62728;">
        \(P(\text{offer} \mid \text{denied the})\) ?
      </div>
    </div>
  </div>
</div>
    <ul style="margin-top: 4px; padding-left: 28px;">
      <li class="fragment" data-fragment-index="4" style="margin-top: 10px;">
        Trigrams can have <b>zero probability</b> (unseen \(n\)-grams ⇒ assign 0).
      </li>
      <li class="fragment" data-fragment-index="5" style="margin-top: 8px;">
        <b>Question:</b> how to avoid these zeros?
      </li>
    </ul>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Smoothing: intuition</div>
  <div class="ppt-line"></div>

  <div style="display:flex; gap:28px; align-items:flex-start; margin-top: 14px;">
    <!-- LEFT: bullets -->
    <div style="flex: 0.95; font-size: 34px; line-height: 1.25; font-weight: 560;">
      <ul style="margin-top: 0; padding-left: 28px;">
        <li class="fragment" data-fragment-index="1">
          When we have <b>sparse statistics</b>
        </li>
      </ul>
      <div class="fragment" data-fragment-index="2" style="margin: 18px 0 10px 8px; font-size: 34px; font-weight: 700;">
        \(q(w \mid \text{denied the})\)
      </div>
      <ul style="margin-top: 10px; padding-left: 28px;">
        <li class="fragment" data-fragment-index="4" style="margin-top: 10px;">
          <b>Steal probability mass</b> to generalize better
        </li>
        <li class="fragment" data-fragment-index="6" style="margin-top: 10px;">
          <b>How to steal?</b>
          <li class="fragment" data-fragment-index="7" style="margin-top: 10px;">
            Additive smoothing<br>Linear interpolation<br>Backoff smoothing (Katz)<br>KN Smoothing Good-Turing<br>
          </li>
        </li>
      </ul>
    </div>
    <!-- RIGHT: charts recreated in pure HTML -->
    <div style="flex: 1.05;">
      <!-- TOP: raw sparse counts -->
      <div class="fragment" data-fragment-index="3" style="
        background: rgba(0,0,0,0.04);
        border: 1px solid rgba(0,0,0,0.12);
        border-radius: 14px;
        padding: 14px 16px;
      ">
        <div style="font-size: 22px; font-weight: 800; margin-bottom: 10px; opacity: 0.85;">
          Raw counts (many zeros)
        </div>
        <!-- bars container -->
        <div style="display:flex; align-items:flex-end; gap:16px; height: 210px;">
          <!-- bar template: height in px, label rotated -->
          <div style="width:60px; display:flex; flex-direction:column; align-items:center;">
            <div style="font-size:22px; font-weight:800; margin-bottom:6px;">3</div>
            <div style="height:170px; width:56px; background: rgba(61,103,199,0.75); border-radius:10px 10px 0 0;
                        display:flex; align-items:center; justify-content:center;">
              <div style="transform: rotate(-90deg); font-size:26px; font-weight:900; color:#111;">
                allegations
              </div>
            </div>
          </div>
          <div style="width:60px; display:flex; flex-direction:column; align-items:center;">
            <div style="font-size:22px; font-weight:800; margin-bottom:6px;">2</div>
            <div style="height:125px; width:56px; background: rgba(61,103,199,0.70); border-radius:10px 10px 0 0;
                        display:flex; align-items:center; justify-content:center;">
              <div style="transform: rotate(-90deg); font-size:26px; font-weight:900; color:#111;">
                reports
              </div>
            </div>
          </div>
          <div style="width:60px; display:flex; flex-direction:column; align-items:center;">
            <div style="font-size:22px; font-weight:800; margin-bottom:6px;">1</div>
            <div style="height:75px; width:56px; background: rgba(61,103,199,0.55); border-radius:10px 10px 0 0;
                        display:flex; align-items:center; justify-content:center;">
              <div style="transform: rotate(-90deg); font-size:24px; font-weight:850; color:#111;">
                claims
              </div>
            </div>
          </div>
          <div style="width:60px; display:flex; flex-direction:column; align-items:center;">
            <div style="font-size:22px; font-weight:800; margin-bottom:6px;">1</div>
            <div style="height:75px; width:56px; background: rgba(61,103,199,0.55); border-radius:10px 10px 0 0;
                        display:flex; align-items:center; justify-content:center;">
              <div style="transform: rotate(-90deg); font-size:24px; font-weight:850; color:#111;">
                request
              </div>
            </div>
          </div>
          <!-- zeros -->
          <div style="width:48px; display:flex; flex-direction:column; align-items:center;">
            <div style="font-size:22px; font-weight:800; margin-bottom:6px; opacity:0.75;">0</div>
            <div style="height:16px; width:44px; background: rgba(0,0,0,0.08); border-radius:8px;"></div>
            <div style="margin-top:8px; font-size:22px; font-weight:800; transform: rotate(-90deg); opacity:0.75;">
              attack
            </div>
          </div>
          <div style="width:48px; display:flex; flex-direction:column; align-items:center;">
            <div style="font-size:22px; font-weight:800; margin-bottom:6px; opacity:0.75;">0</div>
            <div style="height:16px; width:44px; background: rgba(0,0,0,0.08); border-radius:8px;"></div>
            <div style="margin-top:8px; font-size:22px; font-weight:800; transform: rotate(-90deg); opacity:0.75;">
              man
            </div>
          </div>
          <div style="width:48px; display:flex; flex-direction:column; align-items:center;">
            <div style="font-size:22px; font-weight:800; margin-bottom:6px; opacity:0.75;">0</div>
            <div style="height:16px; width:44px; background: rgba(0,0,0,0.08); border-radius:8px;"></div>
            <div style="margin-top:8px; font-size:22px; font-weight:800; transform: rotate(-90deg); opacity:0.75;">
              outcome
            </div>
          </div>
          <div style="font-size:34px; font-weight:900; opacity:0.75; padding-bottom: 10px;">…</div>
        </div>
      </div>
      <!-- BOTTOM: after smoothing (redistribute mass) -->
      <div class="fragment" data-fragment-index="5" style="
        margin-top: 16px;
        background: rgba(0,0,0,0.04);
        border: 1px solid rgba(0,0,0,0.12);
        border-radius: 14px;
        padding: 14px 16px;
      ">
        <div style="display:flex; gap:14px; align-items:flex-start;">
          <!-- small table -->
          <div style="
            flex: 0.85;
            background: rgba(255,255,255,0.65);
            border: 1px solid rgba(0,0,0,0.10);
            border-radius: 12px;
            padding: 10px 12px;
            font-size: 22px;
            line-height: 1.25;
          ">
            <div style="font-weight:900; margin-bottom: 6px;">\(P(w \mid \text{denied the})\)</div>
            <div>2.5 &nbsp; allegations</div>
            <div>1.5 &nbsp; reports</div>
            <div>0.5 &nbsp; claims</div>
            <div>0.5 &nbsp; request</div>
            <div style="margin-top: 6px;">
              <span style="color:#1a7f37; font-weight:900;">2</span>
              &nbsp; <span style="color:#1a7f37; font-weight:900;">other</span>
            </div>
          </div>
          <!-- smoothed bars -->
          <div style="flex: 1.15;">
            <div style="font-size: 22px; font-weight: 800; opacity:0.85; margin-bottom: 8px;">
              After smoothing (some mass for “other”)
            </div>
            <div style="display:flex; align-items:flex-end; gap:14px; height: 160px;">
              <div style="width:46px; height:130px; background: rgba(61,103,199,0.70); border-radius:10px 10px 0 0;"></div>
              <div style="width:46px; height:95px;  background: rgba(61,103,199,0.65); border-radius:10px 10px 0 0;"></div>
              <div style="width:46px; height:55px;  background: rgba(61,103,199,0.50); border-radius:10px 10px 0 0;"></div>
              <div style="width:46px; height:55px;  background: rgba(61,103,199,0.50); border-radius:10px 10px 0 0;"></div>
              <!-- "other" mass shown in green -->
              <div style="width:46px; height:40px; background: rgba(26,127,55,0.70); border-radius:10px 10px 0 0;"></div>
              <div style="font-size:32px; font-weight:900; opacity:0.7; padding-bottom: 6px;">…</div>
            </div>
            <div style="display:flex; gap:14px; margin-top: 10px; font-size: 20px; font-weight: 800; opacity:0.85;">
              <div style="width:46px; text-align:center;">all.</div>
              <div style="width:46px; text-align:center;">rep.</div>
              <div style="width:46px; text-align:center;">cl.</div>
              <div style="width:46px; text-align:center;">req.</div>
              <div style="width:46px; text-align:center; color:#1a7f37;">other</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Additive smoothing (Laplace / Add-\(\delta\))</div>
  <div class="ppt-line"></div>

  <!-- Top: two panels side-by-side -->
  <div style="display:flex; gap:22px; align-items:stretch; margin-top: 10px;">
    <!-- Left-top: Add-one -->
    <div style="
      flex: 1;
      background: rgba(0,0,0,0.05);
      border: 1px solid rgba(0,0,0,0.12);
      border-radius: 14px;
      padding: 14px 16px;
    ">
      <div style="font-size: 30px; font-weight: 900; margin-bottom: 10px;">
        1) Add-one (Laplace) idea
      </div>
      <div style="font-size: 24px; line-height: 1.25;">
        <div style="font-weight: 800; margin-bottom: 6px;">MLE estimate</div>
        \[
          P_{\text{MLE}}(w_i\mid w_{i-1}) =\frac{C(w_{i-1}w_i)}{C(w_{i-1})}
        \]
        <div style="margin-top: 10px; font-weight: 800;">
          <span class="text-green">pretend we saw each word one more time</span> 
        </div>
        \[
          P_{\text{Add}}(w_i\mid w_{i-1}) = \frac{C(w_{i-1}w_i)+1}{C(w_{i-1})+|V|}
        \]
      </div>
    </div>
    <!-- Right-top: Reconstituted counts -->
    <div style="
      flex: 1;
      background: rgba(0,0,0,0.05);
      border: 1px solid rgba(0,0,0,0.12);
      border-radius: 14px;
      padding: 14px 16px;
    ">
      <div style="font-size: 30px; font-weight: 900; margin-bottom: 10px;">
        2) View as pseudo-counts \(C^{*}\)
      </div>
      <div style="font-size: 20px; line-height: 1.25;">
        <div style="margin-bottom: 8px;">
          “Reconstitute” <span class="text-green" style="font-weight: 900;">pseudo-counts</span> from smoothed probabilities.
        </div>
        <div style="font-weight: 800; margin: 8px 0 6px;">Unigrams (train size \(N\))</div>
        \[
          C^{*}(w_i) = P_{\text{Add}}(w_i)\cdot N
          = \frac{C(w_i)+1}{N+|V|}\cdot N
        \]
        <div style="font-weight: 800; margin: 10px 0 6px;">Bigrams</div>
        \begin{align}
          C^{*}(w_{i-1}w_i) &= P_{\text{Add}}(w_i\mid w_{i-1})\cdot C(w_{i-1})\\
          &= \frac{C(w_{i-1}w_i)+1}{C(w_{i-1})+|V|}\cdot C(w_{i-1})
        \end{align}
      </div>
    </div>
  </div>

  <!-- Bottom: Add-delta generalization -->
  <div style="
    margin-top: 18px;
    background: rgba(0,0,0,0.06);
    border: 1px solid rgba(0,0,0,0.12);
    border-radius: 14px;
    padding: 14px 16px;
  ">
    <div style="display:flex; align-items:flex-start; gap:18px;">
      <div style="flex: 1.2;">
        <div style="font-size: 30px; font-weight: 900; margin-bottom: 8px;">
          3) Generalization: Add-\(\delta\) smoothing
        </div>
        <div style="font-size: 24px; line-height: 1.25;">
          Add-one is often too aggressive. Use <span class="text-red" style="font-weight: 900;">Add-\(\delta\)</span>
          (typically \(0<\delta\le 1\)).
        </div>
      </div>
      <div style="
        flex: 1.4;
        background: rgba(255,255,255,0.7);
        border: 1px solid rgba(0,0,0,0.10);
        border-radius: 12px;
        padding: 12px 14px;
        font-size: 26px;
        font-weight: 650;
      ">
        \[
          P_{\text{Add}}(w_i\mid w_{i-1})
          =\frac{C(w_{i-1}w_i)+\delta}{\sum_{w\in V}\big(C(w_{i-1}w)+\delta\big)}
          =\frac{C(w_{i-1}w_i)+\delta}{C(w_{i-1})+\delta|V|}
        \]
        <div style="margin-top: 6px; font-size: 22px; opacity: 0.95;">
          Still simple — better methods (backoff / interpolation) usually work better.
        </div>
      </div>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Additive smoothing example</div>
  <div class="ppt-line"></div>

  <!-- IMPORTANT: relative container so the overlay can position inside -->
  <div style="position: relative;">
    <!-- ===== Row 1: two count tables (normal content) ===== -->
    <div style="display:flex; gap:12px; align-items:stretch; margin-top: 10px;">
      <div style="flex:1; background:rgba(0,0,0,0.05); border:1px solid rgba(0,0,0,0.12);
                  border-radius:14px; padding:15px 14px;">
        <div style="font-size:26px; font-weight:900; margin-bottom:8px;">
          Bigram counts \(C(w_{i-1}w_i)\)
        </div>
        <table style="width:100%; border-collapse: collapse; font-size: 18px;">
          <thead>
            <tr style="background:#3d67c7; color:white;">
              <th style="padding:6px 8px;"></th>
              <th style="padding:6px 8px;">i</th>
              <th style="padding:6px 8px;">want</th>
              <th style="padding:6px 8px;">to</th>
              <th style="padding:6px 8px;">eat</th>
              <th style="padding:6px 8px;">chinese</th>
              <th style="padding:6px 8px;">food</th>
              <th style="padding:6px 8px;">lunch</th>
              <th style="padding:6px 8px;">spend</th>
            </tr>
          </thead>
          <tbody>
            <tr><td style="padding:16px 8px; font-weight:800;">i</td><td style="text-align:center;">5</td><td style="text-align:center;">827</td><td style="text-align:center;">0</td><td style="text-align:center;">9</td><td style="text-align:center;">0</td><td style="text-align:center;">0</td><td style="text-align:center;">0</td><td style="text-align:center;">2</td></tr>
            <tr><td style="padding:16px 8px; font-weight:800;">want</td><td style="text-align:center;">2</td><td style="text-align:center;">0</td><td style="text-align:center;">608</td><td style="text-align:center;">1</td><td style="text-align:center;">6</td><td style="text-align:center;">6</td><td style="text-align:center;">5</td><td style="text-align:center;">1</td></tr>
            <tr><td style="padding:16px 8px; font-weight:800;">to</td><td style="text-align:center;">2</td><td style="text-align:center;">0</td><td style="text-align:center;">4</td><td style="text-align:center;">686</td><td style="text-align:center;">2</td><td style="text-align:center;">0</td><td style="text-align:center;">6</td><td style="text-align:center;">211</td></tr>
            <tr><td style="padding:16px 8px; font-weight:800;">eat</td><td style="text-align:center;">0</td><td style="text-align:center;">0</td><td style="text-align:center;">2</td><td style="text-align:center;">0</td><td style="text-align:center;">16</td><td style="text-align:center;">2</td><td style="text-align:center;">42</td><td style="text-align:center;">0</td></tr>
            <tr><td style="padding:16px 8px; font-weight:800;">chinese</td><td style="text-align:center;">1</td><td style="text-align:center;">0</td><td style="text-align:center;">0</td><td style="text-align:center;">0</td><td style="text-align:center;">0</td><td style="text-align:center;">82</td><td style="text-align:center;">1</td><td style="text-align:center;">0</td></tr>
            <tr><td style="padding:16px 8px; font-weight:800;">food</td><td style="text-align:center;">15</td><td style="text-align:center;">0</td><td style="text-align:center;">15</td><td style="text-align:center;">0</td><td style="text-align:center;">1</td><td style="text-align:center;">4</td><td style="text-align:center;">0</td><td style="text-align:center;">0</td></tr>
            <tr><td style="padding:16px 8px; font-weight:800;">lunch</td><td style="text-align:center;">2</td><td style="text-align:center;">0</td><td style="text-align:center;">0</td><td style="text-align:center;">0</td><td style="text-align:center;">0</td><td style="text-align:center;">1</td><td style="text-align:center;">0</td><td style="text-align:center;">0</td></tr>
            <tr><td style="padding:16px 8px; font-weight:800;">spend</td><td style="text-align:center;">1</td><td style="text-align:center;">0</td><td style="text-align:center;">1</td><td style="text-align:center;">0</td><td style="text-align:center;">0</td><td style="text-align:center;">0</td><td style="text-align:center;">0</td><td style="text-align:center;">0</td></tr>
          </tbody>
        </table>
      </div>
      <div style="flex:1; background:rgba(0,0,0,0.05); border:1px solid rgba(0,0,0,0.12);
                  border-radius:14px; padding:12px 14px;">
        <div style="font-size:26px; font-weight:900; margin-bottom:8px;">
          Laplace counts \(C(w_{i-1}w_i)+1\)
        </div>
        <table style="width:100%; border-collapse: collapse; font-size: 18px;">
          <thead>
            <tr style="background:#3d67c7; color:white;">
              <th style="padding:6px 8px;"></th>
              <th style="padding:6px 8px;">i</th>
              <th style="padding:6px 8px;">want</th>
              <th style="padding:6px 8px;">to</th>
              <th style="padding:6px 8px;">eat</th>
              <th style="padding:6px 8px;">chinese</th>
              <th style="padding:6px 8px;">food</th>
              <th style="padding:6px 8px;">lunch</th>
              <th style="padding:6px 8px;">spend</th>
            </tr>
          </thead>
          <tbody>
            <tr><td style="padding:16px 8px; font-weight:800;">i</td><td style="text-align:center;">6</td><td style="text-align:center;">828</td><td style="text-align:center;">1</td><td style="text-align:center;">10</td><td style="text-align:center;">1</td><td style="text-align:center;">1</td><td style="text-align:center;">1</td><td style="text-align:center;">3</td></tr>
            <tr><td style="padding:16px 8px; font-weight:800;">want</td><td style="text-align:center;">3</td><td style="text-align:center;">1</td><td style="text-align:center;">609</td><td style="text-align:center;">2</td><td style="text-align:center;">7</td><td style="text-align:center;">7</td><td style="text-align:center;">6</td><td style="text-align:center;">2</td></tr>
            <tr><td style="padding:16px 8px; font-weight:800;">to</td><td style="text-align:center;">3</td><td style="text-align:center;">1</td><td style="text-align:center;">5</td><td style="text-align:center;">687</td><td style="text-align:center;">3</td><td style="text-align:center;">1</td><td style="text-align:center;">7</td><td style="text-align:center;">212</td></tr>
            <tr><td style="padding:16px 8px; font-weight:800;">eat</td><td style="text-align:center;">1</td><td style="text-align:center;">1</td><td style="text-align:center;">3</td><td style="text-align:center;">1</td><td style="text-align:center;">17</td><td style="text-align:center;">3</td><td style="text-align:center;">43</td><td style="text-align:center;">1</td></tr>
            <tr><td style="padding:16px 8px; font-weight:800;">chinese</td><td style="text-align:center;">2</td><td style="text-align:center;">1</td><td style="text-align:center;">1</td><td style="text-align:center;">1</td><td style="text-align:center;">1</td><td style="text-align:center;">83</td><td style="text-align:center;">2</td><td style="text-align:center;">1</td></tr>
            <tr><td style="padding:16px 8px; font-weight:800;">food</td><td style="text-align:center;">16</td><td style="text-align:center;">1</td><td style="text-align:center;">16</td><td style="text-align:center;">1</td><td style="text-align:center;">2</td><td style="text-align:center;">5</td><td style="text-align:center;">1</td><td style="text-align:center;">1</td></tr>
            <tr><td style="padding:16px 8px; font-weight:800;">lunch</td><td style="text-align:center;">3</td><td style="text-align:center;">1</td><td style="text-align:center;">1</td><td style="text-align:center;">1</td><td style="text-align:center;">1</td><td style="text-align:center;">2</td><td style="text-align:center;">1</td><td style="text-align:center;">1</td></tr>
            <tr><td style="padding:16px 8px; font-weight:800;">spend</td><td style="text-align:center;">2</td><td style="text-align:center;">1</td><td style="text-align:center;">2</td><td style="text-align:center;">1</td><td style="text-align:center;">1</td><td style="text-align:center;">1</td><td style="text-align:center;">1</td><td style="text-align:center;">1</td></tr>
          </tbody>
        </table>
      </div>
    </div>
    <!-- ===== OVERLAY: Laplace probability table (appears on top) ===== -->
    <div class="fragment" data-fragment-index="2"
         style="
           position: absolute;
           left: 50%;
           top: 0px;                 /* adjust if you want lower */
           transform: translateX(-50%);
           width: 96%;
           z-index: 50;
           background: rgba(255,255,255,0.96);
           border: 2px solid rgba(0,0,0,0.18);
           border-radius: 16px;
           box-shadow: 0 10px 40px rgba(0,0,0,0.20);
           padding: 14px 16px;
         ">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
        <div style="font-size:26px; font-weight:900;">
          Using Laplace smoothing: $P_{\text{Lap}}(w_i\mid w_{i-1})=\frac{C(w_{i-1}w_i)+1}{C(w_{i-1})+|V|}$
        </div>
      </div>
      <div style="overflow:auto; max-height: 560px; margin-top: 8px;">
        <table style="width:100%; border-collapse: collapse; font-size: 18px;">
        <thead>
          <tr style="background:#3d67c7; color:white;">
            <th style="padding:6px 8px;"></th>
            <th style="padding:6px 8px;">i</th>
            <th style="padding:6px 8px;">want</th>
            <th style="padding:6px 8px;">to</th>
            <th style="padding:6px 8px;">eat</th>
            <th style="padding:6px 8px;">chinese</th>
            <th style="padding:6px 8px;">food</th>
            <th style="padding:6px 8px;">lunch</th>
            <th style="padding:6px 8px;">spend</th>
          </tr>
        </thead>
        <tbody>
          <!-- (Example numbers; replace with your computed table if desired) -->
          <tr><td style="padding:16px 8px; font-weight:800;">i</td><td style="text-align:center;">0.0015</td><td style="text-align:center;">0.21</td><td style="text-align:center;">0.00025</td><td style="text-align:center;">0.0025</td><td style="text-align:center;">0.00025</td><td style="text-align:center;">0.00025</td><td style="text-align:center;">0.00025</td><td style="text-align:center;">0.00075</td></tr>
          <tr><td style="padding:16px 8px; font-weight:800;">want</td><td style="text-align:center;">0.0013</td><td style="text-align:center;">0.00042</td><td style="text-align:center;">0.26</td><td style="text-align:center;">0.00084</td><td style="text-align:center;">0.0029</td><td style="text-align:center;">0.0029</td><td style="text-align:center;">0.0025</td><td style="text-align:center;">0.00084</td></tr>
          <tr><td style="padding:16px 8px; font-weight:800;">to</td><td style="text-align:center;">0.00078</td><td style="text-align:center;">0.00026</td><td style="text-align:center;">0.0013</td><td style="text-align:center;">0.18</td><td style="text-align:center;">0.00078</td><td style="text-align:center;">0.00026</td><td style="text-align:center;">0.0018</td><td style="text-align:center;">0.055</td></tr>
          <tr><td style="padding:16px 8px; font-weight:800;">eat</td><td style="text-align:center;">0.00046</td><td style="text-align:center;">0.00046</td><td style="text-align:center;">0.0014</td><td style="text-align:center;">0.00046</td><td style="text-align:center;">0.0078</td><td style="text-align:center;">0.0014</td><td style="text-align:center;">0.02</td><td style="text-align:center;">0.00046</td></tr>
          <tr><td style="padding:16px 8px; font-weight:800;">chinese</td><td style="text-align:center;">0.0012</td><td style="text-align:center;">0.00062</td><td style="text-align:center;">0.00062</td><td style="text-align:center;">0.00062</td><td style="text-align:center;">0.00062</td><td style="text-align:center;">0.052</td><td style="text-align:center;">0.0012</td><td style="text-align:center;">0.00062</td></tr>
          <tr><td style="padding:16px 8px; font-weight:800;">food</td><td style="text-align:center;">0.0063</td><td style="text-align:center;">0.00039</td><td style="text-align:center;">0.0063</td><td style="text-align:center;">0.00039</td><td style="text-align:center;">0.00079</td><td style="text-align:center;">0.002</td><td style="text-align:center;">0.00039</td><td style="text-align:center;">0.00039</td></tr>
          <tr><td style="padding:16px 8px; font-weight:800;">lunch</td><td style="text-align:center;">0.0017</td><td style="text-align:center;">0.00056</td><td style="text-align:center;">0.00056</td><td style="text-align:center;">0.00056</td><td style="text-align:center;">0.00056</td><td style="text-align:center;">0.0011</td><td style="text-align:center;">0.00056</td><td style="text-align:center;">0.00056</td></tr>
          <tr><td style="padding:16px 8px; font-weight:800;">spend</td><td style="text-align:center;">0.0012</td><td style="text-align:center;">0.00058</td><td style="text-align:center;">0.0012</td><td style="text-align:center;">0.00058</td><td style="text-align:center;">0.00058</td><td style="text-align:center;">0.00058</td><td style="text-align:center;">0.00058</td><td style="text-align:center;">0.00058</td></tr>
        </tbody>
      </table>
      </div>
    </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Additive smoothing: reconstituted counts</div>
  <div class="ppt-line"></div>

  <style>
    table.ngram {
      border-collapse: collapse;
      width: 100%;
      font-size: 18px;
    }
    table.ngram th, table.ngram td {
      border: 1px solid rgba(0,0,0,0.18);
      padding: 6px 8px;
      text-align: center;
      vertical-align: middle;
    }
    table.ngram thead th {
      background: #3d67c7;
      color: white;
      font-weight: 900;
    }
    table.ngram tbody th {
      background: rgba(61,103,199,0.10);
      font-weight: 900;
      text-align: left;
      padding-left: 10px;
    }
    .hl {
      outline: 3px solid #d62728;
      outline-offset: -3px;
      border-radius: 6px;
      font-weight: 900;
    }
    .note {
      font-size: 28px;
      font-weight: 900;
      color: #1f6feb;
      margin-top: 10px;
      text-align: center;
    }
  </style>

  <div style="display:flex; gap:22px; align-items:flex-start; margin-top: 12px;">
    <!-- Before -->
    <div style="flex:1; background:rgba(0,0,0,0.05); border:1px solid rgba(0,0,0,0.12);
                border-radius:14px; padding:12px 14px;">
      <div style="font-size: 26px; font-weight: 900; margin-bottom: 8px;">
        Before Add-one smoothing
      </div>
      <table class="ngram">
        <thead>
          <tr>
            <th></th><th>i</th><th>want</th><th>to</th><th>eat</th><th>chinese</th><th>food</th><th>lunch</th><th>spend</th>
          </tr>
        </thead>
        <tbody>
          <tr><th>i</th><td>5</td><td class="hl">827</td><td>0</td><td>9</td><td>0</td><td>0</td><td>0</td><td>2</td></tr>
          <tr><th>want</th><td>2</td><td>0</td><td class="hl">608</td><td>1</td><td>6</td><td>6</td><td>5</td><td>1</td></tr>
          <tr><th>to</th><td>2</td><td>0</td><td>4</td><td>686</td><td>2</td><td>0</td><td>6</td><td>211</td></tr>
          <tr><th>eat</th><td>0</td><td>0</td><td>2</td><td>0</td><td>16</td><td>2</td><td>42</td><td>0</td></tr>
          <tr><th>chinese</th><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>82</td><td>1</td><td>0</td></tr>
          <tr><th>food</th><td>15</td><td>0</td><td>15</td><td>0</td><td>1</td><td>4</td><td>0</td><td>0</td></tr>
          <tr><th>lunch</th><td>2</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td><td>0</td><td>0</td></tr>
          <tr><th>spend</th><td>1</td><td>0</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
        </tbody>
      </table>
    </div>
    <!-- After -->
    <div style="flex:1; background:rgba(0,0,0,0.05); border:1px solid rgba(0,0,0,0.12);
                border-radius:14px; padding:12px 14px;">
      <div style="font-size: 26px; font-weight: 900; margin-bottom: 8px;">
        After Add-one smoothing (reconstituted counts)
      </div>
      <table class="ngram">
        <thead>
          <tr>
            <th></th><th>i</th><th>want</th><th>to</th><th>eat</th><th>chinese</th><th>food</th><th>lunch</th><th>spend</th>
          </tr>
        </thead>
        <tbody>
          <tr><th>i</th><td>3.8</td><td class="hl">527</td><td>0.64</td><td>6.4</td><td>0.64</td><td>0.64</td><td>0.64</td><td>1.9</td></tr>
          <tr><th>want</th><td>1.2</td><td>0.39</td><td class="hl">238</td><td>0.78</td><td>2.7</td><td>2.7</td><td>2.3</td><td>0.78</td></tr>
          <tr><th>to</th><td>1.9</td><td>0.63</td><td>3.1</td><td>430</td><td>1.9</td><td>0.63</td><td>4.4</td><td>133</td></tr>
          <tr><th>eat</th><td>0.34</td><td>0.34</td><td>1</td><td>0.34</td><td>5.8</td><td>1</td><td>15</td><td>0.34</td></tr>
          <tr><th>chinese</th><td>0.2</td><td>0.098</td><td>0.098</td><td>0.098</td><td>0.098</td><td>8.2</td><td>0.2</td><td>0.098</td></tr>
          <tr><th>food</th><td>6.9</td><td>0.43</td><td>6.9</td><td>0.43</td><td>0.86</td><td>2.2</td><td>0.43</td><td>0.43</td></tr>
          <tr><th>lunch</th><td>0.57</td><td>0.19</td><td>0.19</td><td>0.19</td><td>0.19</td><td>0.38</td><td>0.19</td><td>0.19</td></tr>
          <tr><th>spend</th><td>0.32</td><td>0.16</td><td>0.32</td><td>0.16</td><td>0.16</td><td>0.16</td><td>0.16</td><td>0.16</td></tr>
        </tbody>
      </table>
    </div>
  </div>
  <div class="note">Some counts changed too much!</div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Linear interpolation</div>
  <div class="ppt-line"></div>

  <!-- Row 1: bigram interpolation (full width) -->
  <div style="
    background: rgba(0,0,0,0.05);
    border: 1px solid rgba(0,0,0,0.12);
    border-radius: 14px;
    padding: 14px 16px;
    margin-top: 10px;
  ">
    <div style="font-size: 28px; line-height: 1.25; font-weight: 560;">
      <div style="font-weight: 900; font-size: 30px; margin-bottom: 8px;">
        Bigram case
      </div>
      <ul style="margin: 0; padding-left: 26px;">
        <li>
          If some bigrams are unseen (e.g., \(C(\text{burnish the})=0\)), MLE gives zero:
          \(\;P(\text{the}\mid \text{burnish})=0\).
        </li>
        <li style="margin-top: 6px;">
          Combine higher-order and lower-order information via <b>linear interpolation</b> (Mercer, 1980):
        </li>
      </ul>
      <div style="
        margin-top: 10px;
        background: rgba(255,255,255,0.70);
        border: 1px solid rgba(0,0,0,0.10);
        border-radius: 12px;
        padding: 10px 12px;
        font-size: 28px;
        font-weight: 650;
      ">
        \[
          P_{\text{Int}}(w_n\mid w_{n-1})
          = \lambda\cdot P(w_n\mid w_{n-1}) + (1-\lambda)\cdot P(w_n)
        \]
      </div>
    </div>
  </div>
  <!-- Row 2: trigram interpolation + how to set lambdas (side-by-side) -->
  <div style="display:flex; gap:22px; margin-top: 16px; align-items:stretch;">
    <!-- Left: trigram interpolation formula -->
    <div style="
      flex: 1;
      background: rgba(0,0,0,0.05);
      border: 1px solid rgba(0,0,0,0.12);
      border-radius: 14px;
      padding: 14px 16px;
    ">
      <div style="font-size: 30px; font-weight: 900; margin-bottom: 8px;">
        Interpolation for trigram
      </div>
      <div style="font-size: 24px; line-height: 1.25; margin-bottom: 8px;">
        Mix unigram + bigram + trigram (with \(\sum_i \lambda_i=1\)):
      </div>
      <div style="
        background: rgba(255,255,255,0.70);
        border: 1px solid rgba(0,0,0,0.10);
        border-radius: 12px;
        padding: 10px 12px;
        font-size: 26px;
        font-weight: 650;
      ">
        \begin{align}
          P_{\lambda_1,\lambda_2,\lambda_3}(w_n\mid w_{n-2}w_{n-1})
          &= \lambda_1 P(w_n\mid w_{n-2}w_{n-1}) \\
          &+ \lambda_2 P(w_n\mid w_{n-1}) \\
          &+ \lambda_3 P(w_n)
        \end{align}
      </div>
      <div style="margin-top: 10px; font-size: 30px; font-weight: 900; color:#d62728;">
        How to set \(\lambda_i\)?
      </div>
    </div>
    <!-- Right: held-out tuning -->
    <div style="
      flex: 1;
      background: rgba(0,0,0,0.05);
      border: 1px solid rgba(0,0,0,0.12);
      border-radius: 14px;
      padding: 14px 16px;
    ">
      <div style="font-size: 30px; font-weight: 900; margin-bottom: 8px;">
        Tune $\{\lambda_1,\lambda_2,\lambda_3\}$ on held-out data
      </div>
      <div style="
        width: 100%;
        border-radius: 12px;
        overflow: hidden;
        display: flex;
        border: 1px solid rgba(0,0,0,0.12);
        height: 64px;
        margin: 6px 0 12px 0;
      ">
        <div style="flex: 6.5; background:#f2f2f2; display:flex; align-items:center; justify-content:center; font-size: 15px; font-weight:900;">
          Training
        </div>
        <div style="flex: 2.7; background:#e6f2dc; display:flex; align-items:center; justify-content:center; font-size: 15px; font-weight:900;">
          Held-out
        </div>
        <div style="flex: 2.7; background:#dfe9f7; display:flex; align-items:center; justify-content:center; font-size: 15px; font-weight:900;">
          Test
        </div>
      </div>
      <ul style="font-size: 24px; line-height: 1.25; margin: 0; padding-left: 26px;">
        <li>Fix \(N\)-gram probabilities estimated on training data</li>
        <li style="margin-top: 6px;">Choose \(\lambda_i\) to maximize held-out log-likelihood:</li>
      </ul>
      <div style="
        margin-top: 10px;
        background: rgba(255,255,255,0.70);
        border: 1px solid rgba(0,0,0,0.10);
        border-radius: 12px;
        padding: 10px 12px;
        font-size: 26px;
        font-weight: 650;
      ">
        \[
          \max_{\lambda_1,\lambda_2,\lambda_3}\;
          \sum_{i}\log P_{\lambda_1,\lambda_2,\lambda_3}(w_i\mid w_{i-2}w_{i-1})
        \]
      </div>
    </div>

  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Language model toolkits and readings</div>
  <div class="ppt-line"></div>

- When we have sparse statistics

<div class="smoothing-row">
  <div class="smoothing-left">
    $q(\mathbf{w}\mid \text{denied the})$
  </div>

  <div class="smoothing-right">
    <!-- Top bar chart -->
    <svg viewBox="0 0 860 190" width="100%" height="190" role="img" aria-label="Sparse counts">
      <!-- counts on top -->
      <text x="118" y="26" font-size="26" font-weight="700">3</text>
      <text x="228" y="26" font-size="26" font-weight="700">2</text>
      <text x="338" y="26" font-size="26" font-weight="700">1</text>
      <text x="448" y="26" font-size="26" font-weight="700">1</text>
      <text x="558" y="26" font-size="26" font-weight="700">0</text>
      <text x="668" y="26" font-size="26" font-weight="700">0</text>
      <text x="778" y="26" font-size="26" font-weight="700">0</text>
      <text x="828" y="26" font-size="26" font-weight="700">…</text>
      <!-- bars (baseline y=170) -->
      <!-- allegations: 3 -->
      <rect x="90"  y="40" width="70" height="130" fill="var(--bar)" stroke="#111" stroke-width="2"/>
      <!-- reports: 2 -->
      <rect x="200" y="80" width="70" height="90"  fill="var(--bar)" stroke="#111" stroke-width="2"/>
      <!-- claims: 1 -->
      <rect x="310" y="120" width="70" height="50" fill="var(--bar2)" stroke="#111" stroke-width="2"/>
      <!-- request: 1 -->
      <rect x="420" y="120" width="70" height="50" fill="var(--bar2)" stroke="#111" stroke-width="2"/>
      <!-- rotated labels under bars / tokens -->
      <g font-size="28" font-weight="650">
        <text x="120" y="178" transform="rotate(-90 120 178)">allegations</text>
        <text x="235" y="178" transform="rotate(-90 235 178)">reports</text>
        <text x="345" y="178" transform="rotate(-90 345 178)">claims</text>
        <text x="455" y="178" transform="rotate(-90 455 178)">request</text>
        <text x="565" y="178" transform="rotate(-90 565 178)">attack</text>
        <text x="675" y="178" transform="rotate(-90 675 178)">man</text>
        <text x="785" y="178" transform="rotate(-90 785 178)">outcome</text>
        <text x="895" y="178" transform="rotate(-90 845 178)">…</text>
      </g>
    </svg>
  </div>
</div>

- Steal probability mass to generalize better

<div class="smoothing-row" style="margin-top:6px;">
  <div class="smoothing-left prob-block">
    <div style="font-weight:800; margin-bottom:4px;">$P(\mathbf{w}\mid \text{denied the})$</div>
    <div>2.5&nbsp;&nbsp;allegations</div>
    <div>1.5&nbsp;&nbsp;reports</div>
    <div>0.5&nbsp;&nbsp;claims</div>
    <div>0.5&nbsp;&nbsp;request</div>
    <div class="other">2&nbsp;&nbsp;&nbsp;&nbsp;other</div>
  </div>

  <div class="smoothing-right">
    <!-- Bottom bar chart (smoothed) -->
    <svg viewBox="0 0 860 210" width="100%" height="210" role="img" aria-label="Smoothed probabilities">
      <!-- bars (baseline y=190) -->
      <rect x="90"  y="55"  width="70" height="135" fill="var(--bar)"  stroke="#111" stroke-width="2"/>
      <rect x="200" y="85"  width="70" height="105" fill="var(--bar)"  stroke="#111" stroke-width="2"/>
      <rect x="310" y="135" width="70" height="55"  fill="var(--bar2)" stroke="#111" stroke-width="2"/>
      <rect x="420" y="135" width="70" height="55"  fill="var(--bar2)" stroke="#111" stroke-width="2"/>
      <!-- small green mass for "other" spread -->
      <rect x="530" y="170" width="70" height="20" fill="var(--other)" stroke="#111" stroke-width="2"/>
      <rect x="640" y="170" width="70" height="20" fill="var(--other)" stroke="#111" stroke-width="2"/>
      <rect x="750" y="170" width="70" height="20" fill="var(--other)" stroke="#111" stroke-width="2"/>
      <!-- labels -->
      <g font-size="28" font-weight="650">
        <text x="125" y="202" transform="rotate(-90 125 202)">allegations</text>
        <text x="235" y="202" transform="rotate(-90 235 202)">reports</text>
        <text x="345" y="202" transform="rotate(-90 345 202)">claims</text>
        <text x="455" y="202" transform="rotate(-90 455 202)">request</text>
        <text x="565" y="202" transform="rotate(-90 565 202)">attack</text>
        <text x="675" y="202" transform="rotate(-90 675 202)">man</text>
        <text x="785" y="202" transform="rotate(-90 785 202)">outcome</text>
        <text x="845" y="202" transform="rotate(-90 845 202)">…</text>
      </g>
    </svg>
  </div>
</div>

- How to steal?

</section>

---

<section class="ppt">
  <div class="ppt-title">KN smoothing: from Good–Turing to Kneser–Ney</div>
  <div class="ppt-line"></div>
  <!-- Row 1: two panels side-by-side -->
  <div style="display:flex; gap:22px; align-items:stretch; margin-top: 10px;">
    <!-- Left: toy intuition (balls) -->
    <div style="
      flex: 1;
      background: rgba(0,0,0,0.05);
      border: 1px solid rgba(0,0,0,0.12);
      border-radius: 14px;
      padding: 14px 16px;
    ">
      <div style="font-size: 30px; font-weight: 900; margin-bottom: 10px;">
        Good–Turing intuition
      </div>
      <ul style="font-size: 24px; line-height: 1.25; margin: 0; padding-left: 26px;">
        <li>Pick balls from a box (colors = events)</li>
        <li style="margin-top: 6px;">Observed counts:</li>
      </ul>
      <div style="display:flex; gap:14px; margin-top: 10px; font-size: 24px; line-height:1.25;">
        <div style="flex:1;">
          <div>10 red</div>
          <div>3 blue</div>
          <div>2 white</div>
        </div>
        <div style="flex:1;">
          <div style="color:#d62728; font-weight:900;">1 yellow</div>
          <div style="color:#d62728; font-weight:900;">1 black</div>
          <div style="color:#d62728; font-weight:900;">1 green</div>
        </div>
      </div>
      <div style="
        margin-top: 14px;
        background: rgba(255,255,255,0.70);
        border: 1px solid rgba(0,0,0,0.10);
        border-radius: 12px;
        padding: 12px 14px;
        font-size: 24px;
        line-height: 1.25;
        font-weight: 650;
      ">
        Question: how likely is the next ball <span class="text-blue"><b>new color</b></span> (e.g., pink)?
      </div>
      <div style="font-size: 24px; font-weight: 900; margin-bottom: 10px;">
        Frequency-of-frequency
        <div style="margin-bottom: 8px;">
          \(N_r\): number of types seen exactly \(r\) times
        </div>
      </div>
    </div>
    <!-- Right: frequency-of-frequency + r* -->
    <div style="
      flex: 1;
      background: rgba(0,0,0,0.05);
      border: 1px solid rgba(0,0,0,0.12);
      border-radius: 14px;
      padding: 14px 16px;
    ">
      <div style="font-size: 24px; line-height: 1.25;">
        <div style="
          background: rgba(255,255,255,0.70);
          border: 1px solid rgba(0,0,0,0.10);
          border-radius: 12px;
          padding: 10px 12px;
          margin: 10px 0 12px 0;
          font-size: 22px;
        ">
          Example text: <span class="text-red" style="font-weight:900;">Sam I am I am Sam I do not eat</span>
        </div>
        <div style="display:flex; gap:18px; align-items:flex-start;">
          <!-- small unigram table -->
          <div style="flex: 0.9;">
            <table style="width:100%; border-collapse:collapse; font-size: 20px;">
              <thead>
                <tr style="background:#3d67c7; color:white;">
                  <th style="padding:6px 8px; text-align:center;">Unigram</th>
                  <th style="padding:6px 8px; text-align:center;">Count</th>
                </tr>
              </thead>
              <tbody>
                <tr><td style="padding:6px 8px; text-align:center; font-weight:800;">I</td><td style="padding:6px 8px; text-align:center;">3</td></tr>
                <tr><td style="padding:6px 8px; text-align:center; font-weight:800;">Sam</td><td style="padding:6px 8px; text-align:center;">2</td></tr>
                <tr><td style="padding:6px 8px; text-align:center; font-weight:800;">am</td><td style="padding:6px 8px; text-align:center;">2</td></tr>
                <tr><td style="padding:6px 8px; text-align:center; font-weight:800;">do</td><td style="padding:6px 8px; text-align:center;">1</td></tr>
                <tr><td style="padding:6px 8px; text-align:center; font-weight:800;">not</td><td style="padding:6px 8px; text-align:center;">1</td></tr>
                <tr><td style="padding:6px 8px; text-align:center; font-weight:800;">eat</td><td style="padding:6px 8px; text-align:center;">1</td></tr>
              </tbody>
            </table>
          </div>
          <!-- N_r and formulas -->
          <div style="flex: 1.1; font-size: 22px; line-height: 1.25;">
            <div style="margin-top: 2px;">
              \(N_0=3,\; N_1=3,\; N_2=2,\; N_3=1\)
            </div>
            <div style="
              margin-top: 10px;
              background: rgba(255,255,255,0.70);
              border: 1px solid rgba(0,0,0,0.10);
              border-radius: 12px;
              padding: 10px 12px;
              font-size: 22px;
              font-weight: 650;
            ">
              \begin{align}
                & r^{*} =\frac{(r+1)N_{r+1}}{N_r}, \\
                & P^{*}_{\text{GT}}(w) =\frac{r^{*}}{N}
              \end{align}
            </div>
            <div style="margin-top: 8px; color:#1f6feb; font-weight:800;">
              (Show \(P^{*}_{\text{GT}}\) is a distribution!)
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
<div style="display:flex; gap:22px; align-items:stretch; margin-top: 10px;">
    <!-- Left: toy intuition (balls) -->
    <div style="
    margin-top: 16px;
    background: rgba(0,0,0,0.05);
    border: 1px solid rgba(0,0,0,0.12);
    border-radius: 14px;
    padding: 14px 16px;
  ">
    <div style="font-size: 20px; font-weight: 900; margin-bottom: 10px;">
      Kneser–Ney smoothing (idea): From Good–Turing, we often <b>discount</b> observed counts by a constant \(d\),
      then redistribute the leftover mass via interpolation.
    </div>
    <div style="
      margin-top: 10px;
      background: rgba(255,255,255,0.70);
      border: 1px solid rgba(0,0,0,0.10);
      border-radius: 12px;
      padding: 12px 14px;
      font-size: 26px;
      font-weight: 650;
    ">
      \[
        P_{\text{KN}}(w_i\mid w_{i-1})
        = \frac{C(w_{i-1},w_i)-d}{C(w_{i-1})}
        + \lambda(w_{i-1})\cdot P(w_i)
      \]
    </div>
  </div>
    <!-- Right: frequency-of-frequency + r* -->
    <div style="
    margin-top: 16px;
    background: rgba(0,0,0,0.05);
    border: 1px solid rgba(0,0,0,0.12);
    border-radius: 14px;
    padding: 14px 16px;
  ">
    <ul style="font-size: 24px; line-height: 1.25; margin: 10px 0 0 0; padding-left: 26px;">
      <li>\(\frac{C(w_{i-1},w_i)-d}{C(w_{i-1})}\): discounted bigram</li>
      <li>\(\lambda(w_{i-1})\): interpolation weight</li>
    </ul>
  </div>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Quick summary: N-gram language models</div>
  <div class="ppt-line"></div>

  <div style="font-size: 34px; line-height: 1.25; font-weight: 560; margin-top: 16px;">
    <ul style="margin-top: 0; padding-left: 30px;">
      <li class="fragment">
        <b>Idea:</b> approximate next-word probability using only the last \(N-1\) words
        \[
          P(w_t \mid w_{1:t-1}) \approx q(w_t \mid w_{t-N+1:t-1})
        \]
      </li>
      <li class="fragment" style="margin-top: 12px;">
        <b>Training:</b> estimate counts from a corpus (plus smoothing for unseen \(n\)-grams)
      </li>
      <li class="fragment" style="margin-top: 12px;">
        <b>Two major issues</b>
        <ul style="margin-top: 8px; font-size: 30px; line-height: 1.22;">
          <li>
            <b>Parameter explosion:</b> number of \(n\)-grams grows as \(|V|^N\)
            <span style="opacity:0.9;">(e.g., \(|V|=10^4\Rightarrow\) trigram \(\sim 10^{12}\))</span>
          </li>
          <li style="margin-top: 6px;">
            <b>Sparsity / poor generalization:</b> many test \(n\)-grams never appear in training
          </li>
        </ul>
      </li>
      <li class="fragment" style="margin-top: 12px;">
        <b>Today:</b> neural LMs (RNN/Transformer) address these issues via learned representations.
      </li>
    </ul>
  </div>
</section>

---

<section class="ppt">
  <div class="ppt-title">Language model toolkits and readings</div>
  <div class="ppt-line"></div>

  <div style="font-size: 30px; line-height: 1.25; font-weight: 560; margin-top: 14px;">
    <ul style="margin-top: 0; padding-left: 28px;">
      <li style="margin-bottom: 12px;">
        <b>KenLM</b> (fast \(n\)-gram LM toolkit)
        <div style="margin-top: 6px; font-size: 26px;">
          <a href="https://kheafield.com/code/kenlm/" target="_blank" rel="noopener noreferrer"
             style="color:#1f6feb; font-weight:800; text-decoration: underline;">
            https://kheafield.com/code/kenlm/
          </a>
        </div>
      </li>
      <li style="margin-bottom: 12px;">
        <b>Google N-Gram Release (Aug 2006)</b>
        <div style="margin-top: 6px; font-size: 26px;">
          <a href="https://ai.googleblog.com/2006/08/all-our-n-gram-are-belong-to-you.html"
             target="_blank" rel="noopener noreferrer"
             style="color:#1f6feb; font-weight:800; text-decoration: underline;">
            https://ai.googleblog.com/2006/08/all-our-n-gram-are-belong-to-you.html
          </a>
        </div>
        <div style="
          margin-top: 8px;
          background: rgba(0,0,0,0.06);
          border-radius: 12px;
          padding: 10px 12px;
          font-size: 24px;
          line-height: 1.25;
          opacity: 0.95;
        ">
          Tokens: 1,024,908,267,229 · Sentences: 95,119,665,584<br/>
          Unigrams: 13,588,391 · Fivegrams: 1,176,470,663
        </div>
      </li>
      <li style="margin-bottom: 12px;">
        <b>SRILM</b> (classic LM toolkit)
        <div style="margin-top: 6px; font-size: 26px;">
          <a href="http://www.speech.sri.com/projects/srilm/" target="_blank" rel="noopener noreferrer"
             style="color:#1f6feb; font-weight:800; text-decoration: underline;">
            http://www.speech.sri.com/projects/srilm/
          </a>
        </div>
      </li>
      <li style="margin-bottom: 12px;">
        <b>Alias method</b> (fast discrete sampling)
        <div style="margin-top: 6px; font-size: 26px;">
          <a href="https://www.keithschwarz.com/darts-dice-coins/" target="_blank" rel="noopener noreferrer"
             style="color:#1f6feb; font-weight:800; text-decoration: underline;">
            https://www.keithschwarz.com/darts-dice-coins/
          </a>
        </div>
      </li>
      <li style="margin-top: 18px;">
        <b>Reading</b>:
        <span style="color:#1f6feb; font-weight:800;">
          Chapter 4–5. <i>Naive Bayes and Sentiment Classification</i>. <i>Logistic Regression</i>
        </span>
      </li>
    </ul>
  </div>
</section>

