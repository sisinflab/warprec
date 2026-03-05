=================
lastfm-dataset-1K
=================

Version 1.0
March 2010

. What is this?

    This dataset contains <user, artist, song, timestamp> tuples collected from Last.fm API ( http://www.last.fm/api ), 
    using the user.getRecentTracks method ( http://www.last.fm/api/show?service=278 )

    This dataset represents the whole listening habits (till May, 5th 2009) for nearly 1,000 users.

. Data Format:

    The data is formatted one entry per line as follows (tab separated, "\t"):

    userid-timestamp-artid-artname-traid-traname.tsv
      userid \t timestamp \t musicbrainz-artist-id \t artist-name \t musicbrainz-track-id \t track-name

    userid-profile.tsv:
      userid \t gender ('m'|'f'|empty) \t age (int|empty) \t country (str|empty) \t signup (date|empty)

. Example:

    userid-timestamp-artid-artname-traid-traname.tsv:
      user_000639  2009-04-08T01:57:47Z  15676fc4-ba0b-4871-ac8d-ef058895b075  The Dogs D'Amour  6cc252d0-3f42-4fd3-a70f-c8ff8b693aa4  How Do You Fall in Love Again
      user_000639  2009-04-08T01:53:56Z  15676fc4-ba0b-4871-ac8d-ef058895b075  The Dogs D'Amour  aa7dbea2-a0c0-4d0a-9241-5bb98a372b11  Wait Until I'm Dead
      ...

    userid-profile.tsv:
      user_000639     m        Mexico    Apr 27, 2005
      ...

. Data Statistics:

     Total Lines:           19,150,868
     Unique Users:                 992
     Artists with MBID:        107,528
     Artists without MBDID:     69,420

. Files:

    userid-timestamp-artid-artname-traid-traname.tsv (MD5: 64747b21563e3d2aa95751e0ddc46b68)
    userid-profile.tsv                               (MD5: c53608b6b445db201098c1489ea497df)

. License:

    The data in lastfm-dataset-1K is distributed with permission of Last.fm. 
    The data is made available for non-commercial use.
    Those interested in using the data or web services in a commercial context 
    should contact: partners [at] last [dot] fm. 
    For more information see http://www.last.fm/api/tos

. Acknowledgements:

    Thanks to Last.fm for providing the access to this data via their web services. 
    Special thanks to Norman Casagrande.

. Contact:

    This data was collected by Oscar Celma. Send questions or comments to oscar.celma@upf.edu

