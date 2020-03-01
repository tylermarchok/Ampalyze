import itertools
import numpy as np
import spotipy.util
import sklearn.cluster
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def add_uris(fetched):
    for item in fetched['tracks']['items']:
        uris.add(item['track']['uri'])

def features_to_vector(item):
    return np.array([item[key] for key in FEATURE_VECTOR])

# Gets an X-matrix given data as 2-element tuples with IDs and vectors.
def get_x(values):
    return np.vstack([x[1] for x in values])

# Given an object with a .transform(), apply it to the data vectors.
def apply_transform(transformer, data):
    return [(x[0], transformer.transform(x[1].reshape(1, -1))) for x in data]

def train_and_apply(transformer, data):
    X = get_x(data)
    transformer.fit(X)
    return apply_transform(transformer, data)


# privileged_song = input("Enter a song URL bit")
privileged_song = '0UqShk7xMPzDWsJB9s0eFF'

# Create your own Spotify app to get the ID and secret.
# https://beta.developer.spotify.com/dashboard/applications
CLIENT_ID = ''
CLIENT_SECRET = ''

# Put your regular Spotify username here.
USERNAME = ''

REDIRECT_URI = 'https://www.google.com/'
SCOPE = 'user-library-read playlist-modify-public'

# Create a Spotify client that can access my saved song information.
token = spotipy.util.prompt_for_user_token(USERNAME,
                                           SCOPE,
                                           client_id=CLIENT_ID,
                                           client_secret=CLIENT_SECRET,
                                           redirect_uri=REDIRECT_URI)

sp = spotipy.Spotify(auth=token)

# Get the Spotify URIs of each of my saved songs.
uris = set([])


playlist = 'https://open.spotify.com/playlist/3PP4DLeNOFKlK0QwI5P7rf'
results = sp.playlist(playlist)
# results = sp.current_user_saved_tracks()
add_uris(results)
uris.add(privileged_song)

while results['tracks']['next']:
    results = sp.next(results)
    add_uris(results)

# Function that returns the next n elements from the iterator. Used because
# Spotify limits how many items you can group into each of its API calls.
def grouper(n, iterable):
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

# Get the audio features of each of the URIs fetched above.
uris_to_features = {}
for group in grouper(50, uris):
    res = sp.audio_features(tracks=group)
    for item in res:
        uris_to_features[item['uri']] = item

FEATURE_VECTOR = [
    'acousticness',
    'danceability',
    'duration_ms',
    'energy',
    'instrumentalness',
    'key',
    'liveness',
    'loudness',
    'mode',
    'speechiness',
    'tempo',
    'time_signature',
    'valence'
]



vectors = [(x[0], features_to_vector(x[1])) for x in uris_to_features.items()]


scaled = train_and_apply(sklearn.preprocessing.StandardScaler(), vectors)

RUN_ON = scaled

NUM_CLUSTERS = 4
PLAYLIST_NAME_FMT = 'Version {}: Cluster {}'
VERSION = 7

model = sklearn.cluster.KMeans(n_clusters=NUM_CLUSTERS,
                               n_jobs=-1)
model.fit(get_x(RUN_ON))
classified = {}
distance_vals = {}
distance_counts = {}
final_playlist_to_check = -1
final_distance = -1
lowest_dist = 100
highest_dist = 0
playlists = {}
for i in range(0, NUM_CLUSTERS):
    distance_vals[i] = 0
    distance_counts[i] = 0


j=0
for x in RUN_ON:
    rval1, rval2 =  model.predict(x[1])
    playlist = rval1[0]
    distance = rval2[0]

    playlists[j] = playlist
    j+=1

    if(distance > highest_dist):
        highest_dist = distance
    elif(distance < lowest_dist):
        lowest_dist = distance

    classified[x[0]] = model.predict(x[1])
    distance_vals[playlist] += distance
    distance_counts[playlist] += 1
    if(privileged_song in x[0]):
        final_playlist_to_check = playlist
        final_distance = distance


if final_playlist_to_check >= 0:
    avg_dist = (distance_vals[final_playlist_to_check]) / (distance_counts[final_playlist_to_check])
    n1 = round(avg_dist, 2)
    n2 = round(final_distance, 2)
    n3 = round(highest_dist, 2)
    n4 = round(lowest_dist, 2)

    # print("AVG score: {} yours: {} highest: {} lowest {}".format(n1, n2, n3, n4))

    labels = ['AVG score', 'Your score', 'Highest score', 'Lowest score', 'acousticness', 'danceability',
              'duration', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'mode', 'speechiness',
              'tempo', 'time signature', 'valence']


    attrs = []
    for i in range(0, 13):
        # attrs[i] = 0
        attrs.append(0)

    j = 0
    n_songs = 0
    for i in playlists.keys():
        if(playlists[i] == final_playlist_to_check):
            j = 0
            for attr in scaled[i][1][0]:
                attrs[j] += attr
                j+=1
        n_songs += 1


    for j in range(0, 13):
        attrs[j] /= n_songs
        attrs[j] *= 10


    men_means = [n1, n2, n3, n4]

    disp_list = men_means + attrs

    x = np.arange(len(labels))  # the label locations

    plt.figure(figsize=(20, 10))
    plt.bar(x, disp_list, align='center', alpha=0.5)
    plt.xticks(x, labels)
    plt.ylabel('Distance from most \'average\' song in sub-genre')
    plt.title('Song analysis results')

    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(6)

    plt.show()
