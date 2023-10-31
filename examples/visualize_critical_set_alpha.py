
import numpy as np
import plotly.graph_objs as go
import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import diode

import oineus as oin

def sample_circle_random(center, r_min, r_max, num_points):
    # Generate input points sampled from a circle
    theta = np.linspace(0, 2*np.pi, num_points)
    r = np.random.uniform(low=r_min, high=r_max, size=num_input_points)  # Uniform distribution within the circle
    input_points = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
    return input_points


def sample_circle(center, radius, num_points):
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack((x, y))


def sample_epycycles(R=5.0, r_min=0.6, r_max=0.8, num_small_circles=6, num_points=15):
    big_circle_angles = np.linspace(0, 2 * np.pi, num_small_circles, endpoint=False)
    small_circle_centers = np.column_stack((R * np.cos(big_circle_angles), R * np.sin(big_circle_angles)))
    small_circle_radii = np.random.uniform(low=r_min, high=r_max, size=num_small_circles)
    all_points = np.vstack([sample_circle(center, r, num_points) for center, r in zip(small_circle_centers, small_circle_radii)])
    return all_points


num_input_points = 40
avg_r = 1.0
low_r = avg_r - 0.2
high_r = avg_r + 0.2

# input_points = sample_circle_random([0.0, 0.0], low_r, high_r, num_input_points)
# point_cloud_xmin = -2
# point_cloud_xmax = 2
# dgm_xmin = -0.2
# dgm_xmax = 2*avg_r


input_points = sample_epycycles()
point_cloud_xmin = -6
point_cloud_xmax = 6
dgm_xmin = -0.2
dgm_xmax = 8


print(f"{input_points.shape = }")

max_dim = 2
max_radius = 20.0 * high_r
n_threads = 1

init_dim = 1

d_simplices = diode.fill_alpha_shapes(input_points)
oin_simplices = []


for s in d_simplices:
    if len(s[0]) == 1:
        oin_simplices.append(oin.Simplex_double(s[0][0], s[0], s[1]))
    else:
        oin_simplices.append(oin.Simplex_double(s[0], s[1]))

fil = oin.Filtration_double(oin_simplices, negate=False)

top_opt = oin.TopologyOptimizer(fil)
top_opt.reduce_all()
dgms = top_opt.compute_diagram(False)

diagram_points = dgms[init_dim]

print(diagram_points)


def get_simplices(x, y, change_coord, dim, use_critical_sets):
    for pt in dgms.in_dimension(dim, False):
        if np.abs(pt.birth - x) + np.abs(pt.death - y) < 0.0001:
            birth_idx = pt.birth_index
            death_idx = pt.death_index
            break

    if use_critical_sets:
        if change_coord == "inc-death":
            simplex_inds = top_opt.increase_death(death_idx)
        elif change_coord == "dec-death":
            simplex_inds = top_opt.decrease_death(death_idx)
        elif change_coord == "inc-birth":
            simplex_inds = top_opt.increase_birth(birth_idx)
        elif change_coord == "dec-birth":
            simplex_inds = top_opt.decrease_birth(birth_idx)
    else:
        if change_coord in ["inc-death", "dec-death"]:
            simplex_inds = [death_idx]
        elif change_coord in ["inc-birth", "dec-birth"]:
            simplex_inds = [birth_idx]

    simplices = []

    for sigma_idx in simplex_inds:
        sigma_vertices = fil.get_cell(sigma_idx).vertices
        if len(sigma_vertices) == 2:
            simplices.append((input_points[sigma_vertices[0]], input_points[sigma_vertices[1]]))
        elif len(sigma_vertices) == 3:
            simplices.append((input_points[sigma_vertices[0]], input_points[sigma_vertices[1]], input_points[sigma_vertices[2]]))

    return simplices


app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Label('Change direction:'),
            dcc.RadioItems(
                id='change-coord',
                options=[
                    {'label': 'Increase Birth', 'value': 'inc-birth'},
                    {'label': 'Decrease Birth', 'value': 'dec-birth'},
                    {'label': 'Increase Death', 'value': 'inc-death'},
                    {'label': 'Decrease Death', 'value': 'dec-death'},
                ],
                value='inc-birth',
                labelStyle={'display': 'inline-block'}
            ),
        ], style={'display': 'inline-block', 'margin-right': '20px'}),
        html.Div([
            html.Label('Dimension:'),
            dcc.RadioItems(
                id='dimension',
                options=[
                    {'label': '0', 'value': '0'},
                    {'label': '1', 'value': '1'},
                    # {'label': '2', 'value': '2'}
                ],
                value=str(init_dim),
                labelStyle={'display': 'inline-block'}
            ),
        ], style={'display': 'inline-block', 'margin-right': '20px'}),
        html.Div([
            html.Label('Critical set vs diagram method:'),
            dcc.RadioItems(
                id='critical-set',
                options=[
                    {'label': 'Critical set', 'value': 'crit-set'},
                    {'label': 'Diagram method', 'value': 'dgm-method'},
                ],
                value='crit-set',
                labelStyle={'display': 'inline-block'}
            ),
        ], style={'display': 'inline-block', 'margin-right': '20px'}),
        html.Div([
            html.Label('For death/dim 1: triangles or longest edges:'),
            dcc.RadioItems(
                id='plot-chains',
                options=[
                    {'label': 'Triangles', 'value': 'chains'},
                    {'label': 'Edges', 'value': 'edges'},
                ],
                value='chains',
                labelStyle={'display': 'inline-block'}
            ),
        ], style={'display': 'inline-block'}),
    ], style={'margin-bottom': '20px'}),
    html.Div([
        dcc.Graph(
            id='point-cloud',
            figure={
                'data': [go.Scatter(x=input_points[:, 0], y=input_points[:, 1], mode='markers', marker=dict(color='black', size=5))],
                'layout': go.Layout(title='Input Points', xaxis=dict(range=[point_cloud_xmin, point_cloud_xmax]), yaxis=dict(range=[point_cloud_xmin, point_cloud_xmax], scaleanchor="x", scaleratio=1))
            }
            )], style={"width": "49%", "display": "inline-block"}),
    html.Div([
        dcc.Graph(
            id='diagram-plot',
            figure={
                'data': [go.Scatter(x=diagram_points[:, 0], y=diagram_points[:, 1], mode='markers', marker=dict(color='red', size=10))],
                'layout': go.Layout(title='Diagram Points', xaxis=dict(range=[dgm_xmin, dgm_xmax]), yaxis=dict(range=[dgm_xmin, dgm_xmax], scaleanchor="x", scaleratio=1))
            }
        )], style={"width": "49%", "display": "inline-block"}),
])


@app.callback(
    Output('diagram-plot', 'figure'),
    [Input('dimension', 'value')],
    prevent_initial_call=False
)
def update_diagram_points(dimension_value):
    global diagram_points
    # Get the diagram points for the selected dimension
    dim = int(dimension_value)
    diagram_points = dgms[dim]

    # Create a new scatter plot trace for the diagram points
    trace = go.Scatter(
        x=diagram_points[:, 0],
        y=diagram_points[:, 1],
        mode='markers',
        marker=dict(color='red', size=8),
        showlegend=False,
    )
    # Update the figure
    return {
        'data': [trace],
        # 'layout': go.Layout(title='Diagram Points', xaxis=dict(range=[-0.1, 2 * high_r]), yaxis=dict(range=[-0.1, 2 * high_r], scaleanchor="x", scaleratio=1))
        'layout': go.Layout(title='Diagram Points', xaxis=dict(range=[dgm_xmin, dgm_xmax]), yaxis=dict(range=[dgm_xmin, dgm_xmax], scaleanchor="x", scaleratio=1))
    }


@app.callback(
    Output('point-cloud', 'figure'),
    [Input('diagram-plot', 'hoverData'),
     Input('change-coord', 'value'),
     Input('dimension', 'value'),
     Input('critical-set', 'value'),
     Input('plot-chains', 'value')],
    [State('point-cloud', 'figure')],
    prevent_initial_call=True
)
def display_edges(hoverData, change_coord, dimension, crit_or_dgm, plot_chains, existing_figure):
    global dgms
    # Copy the existing figure
    figure = dict(existing_figure)

    # Remove any existing edge traces
    figure['data'] = [trace for trace in figure['data'] if trace['type'] != 'scatter' or 'line' not in trace['mode']]

    dim = int(dimension)

    if hoverData:
        point_index = hoverData['points'][0]['pointIndex']
        x, y = diagram_points[point_index]


        # for 2D data we only need triangles if we modify death value in 1D diagram
        # in all other cases we draw longest edges
        use_critical_sets = crit_or_dgm == "crit-set"
        simplices = get_simplices(x, y, change_coord, dim, use_critical_sets)
        simplex_dim = len(simplices[0])

        if simplices:
            if simplex_dim == 3:
                # Create line segments for the triangles
                line_x = []
                line_y = []
                for triangle in simplices:
                    for i in range(3):  # 3 vertices in a triangle
                        line_x.extend([triangle[i][0], triangle[(i+1)%3][0], None])  # %3 to ensure we cycle back to the first vertex
                        line_y.extend([triangle[i][1], triangle[(i+1)%3][1], None])

                # Add triangles to the input plot
                figure['data'].append(go.Scatter(x=line_x, y=line_y, mode='lines', line=dict(color='blue'), showlegend=False))
            elif simplex_dim == 2:
                for edge in simplices:
                    edge_x, edge_y = zip(*edge)
                    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='blue', width=2), showlegend=False)
                    figure['data'].append(edge_trace)

    return figure

if __name__ == '__main__':
    app.run_server(debug=True)

