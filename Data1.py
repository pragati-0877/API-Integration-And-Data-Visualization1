import matplotlib.pyplot as plt
import networkx as nx
from fpdf import FPDF

# ----------------------------------------------
# 1. Define the Finite State Machine (FSM) Class
# ----------------------------------------------
class FiniteStateMachine:
    def __init__(self, states, alphabet, transitions, start_state, accept_states):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states

    def process_string(self, input_string):
        """Processes a given string and determines if it is accepted by the FSM."""
        current_state = self.start_state
        for symbol in input_string:
            if (current_state, symbol) in self.transitions:
                current_state = self.transitions[(current_state, symbol)]
            else:
                return False  # Invalid transition
        return current_state in self.accept_states

# ----------------------------------------------
# 2. Define a Sample FSM (Example)
# ----------------------------------------------
fsm = FiniteStateMachine(
    states={"q0", "q1", "q2"},
    alphabet={"a", "b"},
    transitions={
        ("q0", "a"): "q1",
        ("q1", "b"): "q2",
        ("q2", "a"): "q1",
    },
    start_state="q0",
    accept_states={"q2"},
)

# ----------------------------------------------
# 3. Generate PDF Report for FSM
# ----------------------------------------------
class AutomataReport:
    def __init__(self, fsm):
        self.fsm = fsm
        self.pdf = FPDF()

    def add_title(self):
        self.pdf.set_font("Arial", "B", 16)
        self.pdf.cell(200, 10, "Finite State Machine Report", ln=True, align="C")
        self.pdf.ln(10)

    def add_fsm_details(self):
        self.pdf.set_font("Arial", "", 12)
        self.pdf.cell(0, 10, f"States: {', '.join(self.fsm.states)}", ln=True)
        self.pdf.cell(0, 10, f"Alphabet: {', '.join(self.fsm.alphabet)}", ln=True)
        self.pdf.cell(0, 10, f"Start State: {self.fsm.start_state}", ln=True)
        self.pdf.cell(0, 10, f"Accept States: {', '.join(self.fsm.accept_states)}", ln=True)
        self.pdf.ln(5)

        self.pdf.cell(0, 10, "Transitions:", ln=True)
        for (state, symbol), next_state in self.fsm.transitions.items():
            self.pdf.cell(0, 10, f"({state}, {symbol}) -> {next_state}", ln=True)
        self.pdf.ln(10)

    def generate_pdf(self, filename="automata_report.pdf"):
        self.pdf.add_page()
        self.add_title()
        self.add_fsm_details()
        self.pdf.output(filename)
        print(f"PDF Report Generated: {filename}")

# ----------------------------------------------
# 4. Visualize FSM using NetworkX & Matplotlib
# ----------------------------------------------
def draw_fsm(fsm):
    G = nx.DiGraph()
    
    for (state, symbol), next_state in fsm.transitions.items():
        G.add_edge(state, next_state, label=symbol)

    pos = nx.spring_layout(G)  # Better visualization
    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}

    # Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=3000, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    plt.title("Finite State Machine Diagram")
    plt.show()

# ----------------------------------------------
# 5. Run the Program (Generate Report & Draw FSM)
# ----------------------------------------------
if __name__ == "__main__":
    # Generate PDF Report
    report = AutomataReport(fsm)
    report.generate_pdf()

    # Visualize the FSM
    draw_fsm(fsm)
