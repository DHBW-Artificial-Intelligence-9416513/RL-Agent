# Implementierung eines RL-Agenten zum Lösen von einer Gym-Umgebung

---

### Gliederung der Dokumentation

1. [Beschreibung des Projekts](#beschreibung-des-projekts)
2. [Author und Kontaktinformationen des Erstellers](#author-und-kontaktinformationen-des-erstellers)
3. [Zielsetzung des Projekts](#zielsetzung-des-projekts)
4. [Beschreibung der Gym-Umgebung](#beschreibung-der-gym-umgebung)
5. [Aufbau und Verwendung des Projekts](#aufbau-und-verwendung-des-projekts)
    1. [Kopieren des Projekts](#kopieren-des-projekts)
    2. [Aufbau des Projekts](#aufbau-des-projekts)
    3. [Installation der Abhängigkeiten des Projekts](#installation-der-abhängigkeiten-des-projekts)
        1. [Installation mit pip](#installation-mit-pip)
        2. [Installation mit conda](#installation-mit-conda)
        3. [Installation mit venv](#installation-mit-venv)
    4. [Ausführen des Projekts](#ausführen-des-projekts)
6. [Durchführung des Projekts](#durchführung-des-projekts)
7. [Ergebnisse des Projekts](#ergebnisse-des-projekts)
    1. [Visualisierung der Ergebnisse](#visualisierung-der-ergebnisse)
    2. [Bewertung der Ergebnisse](#bewertung-der-ergebnisse)
8. [Fazit des Projekts](#fazit-des-projekts)
9. [Quellen](#quellen)
10. [Lizenz](#lizenz)

---

## Beschreibung des Projekts

In der Vorlesung "Weiterführende Aspekte der Künstlichen Intelligenz" wurde das Thema Reinforcement Learning behandelt.
Um ein besseres Verständnis, sowie eine grundlegen Praxiserfahrung mit dem Thema zu erlangen, wird in dieses Projekt ein
RL-Agent implementiert, der in der Lage ist, eine Gym-Umgebung zu lösen. Dabei soll der Agent möglichst eine hohe
Punktzahl (Reward) erreichen. Als Sekundärziel soll der Agent mithilfe von Torch implementiert werden. Als Gym-Umgebung
wurde sich für die Umgebung **[LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)**
entschieden. Diese Umgebung wird zu Beginn des Projekts näher beschrieben. und anschließend werden die Ergebnisse des
RL-Agents, für eine jeweilige Trainingsepisode Menge vorgestellt.

---

## Author und Kontaktinformationen des Erstellers

Dieses Projekt wurde von **[Tobias Kister (9416513)](mailto:contact@tkister.de)**, im Rahmen der Vorlesung
"Weiterführende Aspekte der Künstlichen Intelligenz" als eine Abgabe für den Themenbereich 3 - Reinforcement Learning
selbständig entwickelt und eingereicht.

---

## Zielsetzung des Projekts

Die Zielsetzung des Projekts ist es, einen RL-Agenten zu implementieren, der in der Lage ist, die Gym-Umgebung
**[LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)** zu lösen. Dabei soll der Agent
möglichst
eine hohe Punktzahl (Reward)
erreichen. Als Sekundärziel wollte ich mich explizit mit der Implementierung von *DQN-Agenten* beschäftigen und diese
durch **[Torch](https://pytorch.org/)** umsetzen.

---

## Beschreibung der Gym-Umgebung

Im Rahmen des Projekts habe ich mich für die Gym-Umgebung
**[LunarLander](https://gymnasium.farama.org/environments/box2d/lunar_lander/)** entschieden.
Ziel dieser Umgebung ist es, einen Lander auf dem Mond zu landen. Dabei muss der Lander möglichst sanft auf dem Boden
aufsetzen und darf nicht zu schnell fliegen. Zudem muss er es ohne viele Zündungen erreichen, ebenso soll er zwischen
zwei Fahnen landen. Der Lander kann durch die folgenden Aktionen gesteuert werden:

| Index | Beschreibung         |
|-------|----------------------|
| 0     | Nichts tun           |
| 1     | Linken Motor zünden  |
| 2     | Hauptmotor zünden    |
| 3     | Rechten Motor zünden |

Damit handelt es sich um eine diskrete Actionsspace. Der Zustand des Lander wird durch einen 8-dimensionalen Vektor
beschrieben, welcher laut der Dokumentation folgende Werte enthält:

| Index | Beschreibung                                     |
|-------|--------------------------------------------------|
| 0     | x-Position des Lander                            |
| 1     | y-Position des Lander                            |
| 2     | x-Geschwindigkeit des Lander                     |
| 3     | y-Geschwindigkeit des Lander                     |
| 4     | Winkel des Lander                                |
| 5     | Winkelgeschwindigkeit des Lander                 |
| 6     | True, wenn linkes Bein Kontakt hat, sonst False  |
| 7     | True, wenn rechtes Bein Kontakt hat, sonst False |

Damit handelt es sich um einen kontinuierlichen Zustandsraum. Die Umgebung ist gelöst, wenn der Lander mit einem Score
von 200 oder mehr landet. Um diese Punktzahl zu erreichen, muss der Länder sanft auf dem Boden aufsetzen, ohne zu allzu
häufig die Triebwerke zu zünden.

<p align="center">
Darstellung der Umgebung: LunarLander
</p>
<p align="center">
  <img width="460" height="300" src="https://github.com/DHBW-Artificial-Intelligence-9416513/RL-Agent/blob/4643ce51d1e685bbbd021aea8a61b9a68d0f12f8/data/lunar_lander_doku.gif">
</p>
<p align="center">
Diese Darstellung wurde aus der Dokumentation der Umgebung entnommen.
</p>



---

## Aufbau und Verwendung des Projekts

Bevor nun die eigentliche Implementierung des Projekts beschrieben wird, soll zunächst der Aufbau des Projekts, sowie
die Abhängigkeiten und die Verwendung des Projekts beschrieben werden.

### Kopieren des Projekts

Um das Projekt selber in einer beliebigen IDE zu öffnen, muss das Projekt zunächst geklont werden. Dazu muss folgende
Befehl in einen Terminal eingegeben werden:

```shell
git clone https://github.com/DHBW-Artificial-Intelligence-9416513/RL-Agent.git
```

### Aufbau des Projekts

Nach den erfolgreichen Klonen des Projekts sollte sich nun ein Ordner mit dem Namen **RL-Agent** im aktuellen
Verzeichnis des Terminals befinden. In diesem Ordner befindet sich jetzt das Projekt. Der Aufbau des Projekts ist wie
folgt:

```text
|- RL-Agent
|  |- data
|  |  |- img_lunarlander
|  |  |- videos_lunarlander
|  |  |- .gitkeep
|  |  |- lunar_lander_doku.gif
|  |- models
|  |  |- acrobot
|  |  |  |- policy_net.pt
|  |  |  |- target_net.pt
|  |  |- lunarlander
|  |  |  |- DQN_MODEL_100.pth
|  |  |  |- DQN_MODEL_250.pth
|  |  |  |- DQN_MODEL_500.pth
|  |  |  |- DQN_MODEL_750.pth
|  |  |  |- DQN_MODEL_1000.pth
|  |  |- .gitkeep
|  |- notebooks
|  |  |- Hand-on-Coding-Gymnasium-Acrobot.ipynb
|  |  |- LunarLander.ipynb
|  |  |- .gitkeep
|  |- .gitignore
|  |- LICENSE
|  |- README.md
|  |- requirements.txt

```

Der Ordner **data** enthält alle Daten, die während der Ausführung des Projekts anfallen. Der Ordner **models** enthält
alle trainierten Modelle, welche durch die Notebooks in **notebooks** erstellt wurden, oder als Pretrained
heruntergeladen
worden sind.

### Installation der Abhängigkeiten des Projekts

Bevor man das Projekt ausführen kann, müssen zunächst die Abhängigkeiten des Projekts installiert werden. Dazu gibt es
die folgenden Möglichkeiten: **[pip](https://pip.pypa.io/en/stable/)**, **[conda](https://docs.conda.io/en/latest/)**
und **[venv](https://docs.python.org/3/library/venv.html)**.
Alle drei Möglichkeiten werden im Folgenden beschrieben, setzen jedoch eine installierte Python-Version voraus (Dieses
Projekt wurde mit Python 3.11.* entwickelt). Sollte noch keine passende Python-Version installiert sein, kann diese von
der offiziellen **[Python-Website](https://www.python.org/downloads/)** heruntergeladen werden.

**Beachte:** Es gibt neben den Abhängigkeiten des Projekts noch weitere Abhängigkeiten, welche durch das Operating
System definiert werden. Bitte dazu entsprechend die Fehlermeldungen beachten.

#### Installation mit pip

Um die Abhängigkeiten mit pip zu installieren, muss folgender Befehl in ein Terminal eingegeben werden:

```shell
pip install -r requirements.txt
```

**Beachte:** Durch diese Methode werden die Pakete global installiert. Sollte dies nicht gewünscht sein, sollte eine der
anderen Methoden verwendet werden.

#### Installation mit conda

Bevor man die einzelnen Abhängigkeiten installieren kann, muss zunächst conda installiert werden und aktiviert sein.
Beachten Sie dabei bitte die offizielle Dokumentation von **[conda](https://docs.conda.io/en/latest/)**. Nachdem conda
eingerichtet worden ist, bietet es sich an für dieses Projekt eine eigene Environment zu erstellen, um dort nur die
benötigten Bibliotheken zu installieren. Führen Sie dazu die folgenden Schritte durch:

```shell
conda create --name <Name der Environment> python=3.11
```

Nach der erstellung der Environment, muss diese zunächst durch den nachfolgenden Schritt aktiviert werden.

```shell
conda activate <Name der Environment>
```

**Beachte:** Sollte die Environment nicht mehr benötigt werden, kann diese durch den folgenden Befehl wieder deaktiviert
werden. Zudem kann es erforderlich sein, dass bei jeder neuen Sitzung die Environment erneut aktiviert werden muss.

```shell
conda deactivate
```

Nachdem die Environment aktiviert worden ist, können nun die Abhängigkeiten installiert werden. Dazu muss folgender
Befehl in ein Terminal (mit aktivierter Environment) eingegeben werden:

```shell
conda install --file requirements.txt
```

#### Installation mit venv

Um die Abhängigkeiten mit venv zu installieren, muss zunächst ebenfalls eine neue Environment durch den folgenden Befehl
in einem Terminal erstellt werden:

```shell
python -m venv <Name der Environment>
```

Nachdem die Environment erstellt worden ist, muss diese zunächst durch den nachfolgenden Schritt aktiviert werden.

```shell
<Name der Environment>\Scripts\activate.bat
```

**Beachte:** Sollte die Environment nicht mehr benötigt werden, kann diese durch den folgenden Befehl wieder deaktiviert
werden. Zudem kann es erforderlich sein, dass bei jeder neuen Sitzung die Environment erneut aktiviert werden muss.

```shell
deactivate
```

Nachdem die Environment aktiviert worden ist, können nun die Abhängigkeiten installiert werden. Dazu muss folgender
Befehl in ein Terminal (mit aktivierter Environment) eingegeben werden:

```shell
pip install -r requirements.txt
```

Damit sind nun alle Abhängigkeiten installiert und das Projekt kann ausgeführt werden.

### Ausführen des Projekts

Um das Projekt auszuführen, muss zunächst ein Jupiter Notebook Server gestartet werden. Dazu muss folgender Befehl in
ein Terminal, welches entweder eine aktive Python-Umgebung oder eine aktive conda-Umgebung besitzt, eingegeben werden:

```shell
jupyter notebook
```

Anschließend kann über den Browser auf die Notebooks zugegriffen werden. Dazu muss in der Adresszeile des Browsers
folgender Link eingegeben werden:

```text
http://localhost:8888/tree
```

Nachdem der Link aufgerufen worden ist, sollte sich ein Fenster mit dem Inhalt des Notebooks öffnen. Nun kann das
Notebook ausgeführt werden.

---

## Durchführung des Projekts

Da ich zunächst noch keine Erfahrungen in der Entwicklung von Reinforcement Learning Algorithmen hatte und zudem noch
nie PyTorch für die Entwicklung von Machine-Learning Projekten verwendet habe, habe ich mich zunächst mit den Grundlagen
von Reinforcement Learning und PyTorch auseinandergesetzt. Dazu habe ich zunächst mit einem einfachen Beispiel und unter
Verwendung der PyTorch Dokumentation/Tutorial ein überführung in meine Umgebung durchgeführt. Dabei habe ich den Code
von den folgenden Quellen verwendet: [2],[3] Anders als jedoch in den Quellen, habe ich mich dazu entschieden, die
Environment **['Acrobot-v1'](https://gymnasium.farama.org/environments/classic_control/acrobot/)** zu verwenden. Diese
Environment ist ein einfaches Beispiel, welches sich gut für das
Erlernen der Grundlagen geeignet hat. Der entsprechende Code kann in dem folgenden Notebook gefunden
werden: [Hand-on-Coding-Gymnasium-Acrobot.ipynb](notebooks/Hand-on-Coding-Gymnasium-Acrobot.ipynb)

Nachdem ich nun ein besseres Verständnis für die Implementierung von Reinforcement Learning Algorithmen mit PyTorch
hatte, habe ich daran gesetzt das Projekt zu lösen.

Zunächst musste ich in Erfahrung bringen, wie die Environment *
*['LunarLander-v2'](https://gymnasium.farama.org/environments/box2d/lunar_lander/)** funktioniert. Dazu habe ich mir die
Dokumentation der Environment durchgelesen und mir die entsprechenden Methoden und Attribute angeschaut. Zudem habe ich
mir eine mögliche
Lösung ([Chanseok Kang](https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/07/DQN-LunarLander.html))
angeschaut, um ein besseres Verständnis für die Environment zu bekommen. Nachdem ich nun ein besseres Verständnis für
die Environment hatte, habe ich mich daran gesetzt, die entsprechenden Klassen für das Projekt zu entwickeln. Dabei habe
ich mich an der Dokumentation von PyTorch, sowie den Tutorial orientiert und die entsprechenden Klassen entwickelt. Der
Code kann in dem folgenden Notebook gefunden werden: [LunarLander-2.0.ipynb](notebooks/LunarLander-2.0.ipynb) (
**Beachte:** Der Code stammt teilweise aus der Quelle [4])

---

## Ergebnisse des Projekts

Im Folgenden werden die Ergebnisse des Projekts dargestellt. Dabei wird zunächst auf die Ergebnisse des Modells für die
Umgebung **['LunarLander-v2'](https://gymnasium.farama.org/environments/box2d/lunar_lander/)** eingegangen.
Grundsätzlich lässt sich sagen, dass mein trainiertes Modell besser abschneidet als das Original
von [Chanseok Kang](https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/07/DQN-LunarLander.html).
Das lässt sich sagen, da mein Modell schneller einen Average-Score größer als 200 erreicht hat. Nahezu 10% (ca. 175
Episoden) früher als das Vergleichsmodell. Zudem ist mein Modell in der Lage einen Score von 300 zu erreichen, was das
Vergleichsmodell nicht schafft.
Die einzelnen Ergebnisse können in der nachfolgenden Visualisation eingesehen werden.

### Visualisierung der Ergebnisse

In der nachfolgenden Tabelle werden die Ergebnisse des Modells für die Umgebung
**['LunarLander-v2'](https://gymnasium.farama.org/environments/box2d/lunar_lander/)** für jeweilige Trainingsepochen
dargestellt.
Man findet in der Tabelle die Lernkurve des Modells, sowie ein Video des Modells, welches die Validierung des Modells
darstellt. Es ist jedoch zu beachten, dass die Videos nur eine Epoche abbilden und somit nicht die gesamte Qualität des
Modells darstellen müssen.

| Lernfortschritt                                                                                                                                                                                                                                                                                                                                                       | Video des Modells zur Validierung                                                                 |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| ![alt](D:\DHBW\JetBrains\RL-Agent\data\img_lunarlander\DQN_MODEL_100.png "Darstellung des Lernvorschritts in den ersten 100 Episoden")<br/>Darstellung des Lernvorschritts in den ersten 100 Episoden. Es ist ersichtlich, dass kein Score über 0 gekommen ist, was ein Indiz dafür ist, dass es noch keine erfolgreiche Simulationen gab.                            | ![LunarLander-v2_DQN_MODEL_100.gif](data%2Fvideos_lunarlander%2FLunarLander-v2_DQN_MODEL_100.gif) |
| ![alt](D:\DHBW\JetBrains\RL-Agent\data\img_lunarlander\DQN_MODEL_250.png "Darstellung des Lernvorschritts in den ersten 250 Episoden")<br/>Darstellung des Lernvorschritts in den ersten 250 Episoden. Der Score nähert sich langsam den Null-Wert an, was bedeutet, dass die Strafen sich bei den Agenten reduziert hat.                                             | ![LunarLander-v2_DQN_MODEL_250.gif](data%2Fvideos_lunarlander%2FLunarLander-v2_DQN_MODEL_250.gif) |
| ![alt](D:\DHBW\JetBrains\RL-Agent\data\img_lunarlander\DQN_MODEL_500.png "Darstellung des Lernvorschritts in den ersten 500 Episoden")<br/>Darstellung des Lernvorschritts in den ersten 500 Episoden. Es kam zu einzelnen erfolgreichen Simulationen (da Score Teilweise über 200), jedoch ist erkennbar das der Score ab Episode 450 wieder abnahm.                 | ![LunarLander-v2_DQN_MODEL_500.gif](data%2Fvideos_lunarlander%2FLunarLander-v2_DQN_MODEL_500.gif) |
| ![alt](D:\DHBW\JetBrains\RL-Agent\data\img_lunarlander\DQN_MODEL_750.png "Darstellung des Lernvorschritts in den ersten  Episoden")<br/>Darstellung des Lernvorschritts in den ersten 250 Episoden. Ab Episode 686 erzielt der Agent ein Durchschnitt-Score von über 200. Wodurch sich sagen lässt, dass der Agent nun verstanden hat, wie er den Lander landen kann. | ![LunarLander-v2_DQN_MODEL_500.gif](data%2Fvideos_lunarlander%2FLunarLander-v2_DQN_MODEL_750.gif) |                                                                                                                                                                                                                                                                                                                                                              |                                   |

---

## Fazit des Projekts

Durch das Projekt konnte ich ein besseres Verständnis für die Entwicklung von Reinforcement Learning Algorithmen
entwickeln, im Besonderen für DQN-Netzwerken. Zudem konnte ich mich mit der PyTorch Bibliothek auseinandersetzen und
einige praktische Übungen durchführen. Das Projekt hat mir gezeigt, dass es nicht immer einfach ist, ein Modell zu
entwickeln und die Umgebungen auch kein einheitliches Verhalten aufweisen. So ist es mir nicht gelungen, ein Modell für
die Umgebung **['Blackjack-v1'](https://gymnasium.farama.org/environments/toy_text/blackjack/)** zu entwickeln. Dies
liegt daran, dass die Umgebung nicht einheitlich ist und somit das Modell nicht in der Lage ist, ein einheitliches
Verhalten zu erlernen. Dieses Problem ist auch in der Literatur bekannt und wird als **Sparse Rewards** bezeichnet.

Wenn man an den Ergebnissen des Modells für die Umgebung etwas optimieren möchte, so könnte man die Hyperparameter
anpassen oder aber ein anderer RL-Algorithmus, wie zum Beispiel durch Proximal Policy Optimization (PPO) lösen. Eine
mögliche Lösung kann unter den [folgenden Linkfolgenden Link](https://pylessons.com/LunarLander-v2-PPO) gefunden werden.

---

## Quellen

Das Projekt wurde mit folgenden Quellen entwickelt:

- [1] [OpenAI Gym](https://gym.openai.com/)
- [2] [Gymnasium](https://gymnasium.farama.org/)
- [3] [PyTorch DQN-Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [4] [Deep Q-Network (DQN) on LunarLander-v2](https://goodboychan.github.io/python/reinforcement_learning/pytorch/udacity/2021/05/07/DQN-LunarLander.html)

---

## Lizenz

Dieses Projekt ist lizenziert unter der MIT-Lizenz - siehe die [LICENSE](LICENSE) Datei für Details.
