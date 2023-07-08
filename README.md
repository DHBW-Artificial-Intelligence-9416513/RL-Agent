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

---

## Author und Kontaktinformationen des Erstellers

Dieses Projekt wurde von **[Tobias Kister (9416513)](mailto:contact@tkister.de)**, im Rahmen der Vorlesung
"Weiterführende Aspekte der Künstlichen Intelligenz" als eine Abgabe für den Themenbereich 3 - Reinforcement Learning
selbständig entwickelt und eingereicht.

---

## Zielsetzung des Projekts

Die Zielsetzung des Projekts ist es, einen RL-Agenten zu implementieren, der in der Lage ist, die Gym-Umgebung *
*[<Name der Umgebung>](<Link zur Umgebung>)** zu lösen. Dabei soll der Agent möglichst eine hohe Punktzahl (Reward)
erreichen. Als Sekundärziel wollte ich mich explizit mit der Implementierung von *DQN* und *DDQN* beschäftigen und diese
durch **[Torch](https://pytorch.org/)** umsetzen.

---

## Beschreibung der Gym-Umgebung

Im Rahmen des Projekts habe ich mich für die Gym-Umgebung **[<Name der Umgebung>](<Link zur Umgebung>)** entschieden.

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
|  |  |- .gitkeep
|  |- models
|  |  |- acrobot
|  |  |  |- policy_net.pt
|  |  |  |- target_net.pt
|  |  |- lunarlander
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
Environment **['Acrobot-v1'](https://gymnasium.farama.org/environments/classic_control/acrobot/)** zu verwenden. Diese Environment ist ein einfaches Beispiel, welches sich gut für das
Erlernen der Grundlagen geeignet hat. Der entsprechende Code kann in der folgenden Notebook gefunden werden: [Hand-on-Coding-Gymnasium-Acrobot.ipynb](notebooks/Hand-on-Coding-Gymnasium-Acrobot.ipynb)

Nachdem ich nun ein besseres Verständnis für die Implementierung von Reinforcement Learning Algorithmen mit PyTorch
hatte, habe ich daran gesetzt das Projekt zu lösen.

---

## Ergebnisse des Projekts

### Visualisierung der Ergebnisse

### Bewertung der Ergebnisse

---

## Fazit des Projekts

---

## Quellen

Das Projekt wurde mit folgenden Quellen entwickelt:

- [1] [OpenAI Gym](https://gym.openai.com/)
- [2] [Gymnasium](https://gymnasium.farama.org/)
- [3] [PyTorch DQN-Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)

---

## Lizenz

Dieses Projekt ist lizenziert unter der MIT-Lizenz - siehe die [LICENSE](LICENSE) Datei für Details.
