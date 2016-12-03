# Neuron output

$$b_i=\varphi(\Sigma_j w_{ij} a_j)$$


# Idee Backpropagation

* Gegeben Fehlerfunkion $E(w)$, die den Fehler für ein Gewicht $w$ misst und ein konkretes Gewicht $w_{(t)}$
* Verbesserungswert für das Gewicht $w_{(t)}$ mit Hilfe des Gradienten berechnen:
  $$\Delta w_{(t)} := \alpha\Delta w_{(t-1)}-\epsilon\frac{\partial E}{\partial w}(w_{(t)})$$
* Alternativ besser Nesterov Update (richtig?):
  $$\Delta w_{(t)} := \alpha\Delta w_{(t-1)}-\epsilon\frac{\partial E}{\partial w}(w_{(t)}+\alpha\Delta w_{(t-1)})$$
* Das verbesserte Gewicht $w_{(t)}$ ergibt sich dann zu:
  $$w_{(t+1)} := w_{(t)} + \Delta w_{(t)}$$ 

## Outputknoten

* Mean square error (MSE) Outputknoten:
  $$MSE=\frac{1}{n}\Sigma_{i=0}^{n-1}(t_i - \varphi(b_i))^2$$
  $$=\frac{1}{n}(t_0 - \varphi(w_{00}a_0 + w_{01}a_1+\ldots))^2+\frac{1}{n}(t_1 - \varphi(w_{10}a_0 + w_{11}a_1+\ldots))^2+\ldots$$ 
* Gradient für $w_{ij}$:
  $$\frac{\partial E}{\partial w_{ij}}=\frac{-2}{n}(t_i-b_i)\varphi'(b_i)a_j$$
  $$\Delta w_{(t)} = \alpha\Delta w_{(t-1)}+\epsilon\frac{2}{n}(t_i-b_i)\varphi'(b_i)a_j$$
* Alternativ Cross Entropy Error Function (CE):
  $$CE=-\frac{1}{n}\Sigma_{i=0}^{n-1}(t_i\ln(b_i)+(1-t_i)\ln(1-b_i))$$
  * nur für $sigmoid$ sinnvoll?

