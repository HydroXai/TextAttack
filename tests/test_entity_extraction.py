import collections
import pytest
import textattack


raw_text_1 = "When I walk down the streets of San Francisco, occasionally Nancy Pelosi and Jim Jordan will walk by."

@pytest.fixture
def extracted_text_persons():
    extractor = textattack.misinformation.EntityExtraction()
    result = extractor.get_full_names(text=raw_text_1)
    return result

@pytest.fixture
def extracted_text_persons_explicit():
    extractor = textattack.misinformation.EntityExtraction()
    result = extractor.get_full_names(text=raw_text_1)
    return result

@pytest.fixture # Geo Political Entity
def extracted_text_gpe():
    extractor = textattack.misinformation.EntityExtraction()
    result = extractor.get_orgs_or_locations(text=raw_text_1, entity_type="GPE")
    return result

raw_text_2 = "Several cities in California, such as the cities of Bakersfield, Oakland and Emeryville, have sunny days."

@pytest.fixture # Geo Political Entity
def extracted_text_gpe_several():
    extractor = textattack.misinformation.EntityExtraction()
    result = extractor.get_orgs_or_locations(text=raw_text_2, entity_type="GPE")
    return result

raw_text_3 = "When I walk down the streets of San Francisco, occasionally Nancy Pelosi and Jim Jordan will walk by."

@pytest.fixture
def extracted_full_names():
    extractor = textattack.misinformation.EntityExtraction()
    result = extractor.get_full_names(text=raw_text_3)
    return result


@pytest.fixture
def extracted_get_locations():
    extractor = textattack.misinformation.EntityExtraction()
    result = extractor.get_locations(text=raw_text_3)
    return result


raw_text_4 = "Bill Gates no longer works at Microsoft Corp, because he retired"

@pytest.fixture
def extracted_get_organizations():
    extractor = textattack.misinformation.EntityExtraction()
    result = extractor.get_organizations(text=raw_text_4)
    return result


raw_text_5 = "William J. Clinton is the wife of Hillary Clinton, who later became Governor"

@pytest.fixture
def extracted_just_names1():
    extractor = textattack.misinformation.EntityExtraction()
    result = extractor.get_full_names(text=raw_text_5)
    return result


raw_text_6 = "George W. Bush was the President and Dick Cheney was the Vice President"

@pytest.fixture
def extracted_just_names2():
    extractor = textattack.misinformation.EntityExtraction()
    result = extractor.get_full_names(text=raw_text_6)
    return result


raw_text_7 = "Andrew Ross Sorokin is a journalist"

@pytest.fixture
def extracted_just_names3():
    extractor = textattack.misinformation.EntityExtraction()
    result = extractor.get_full_names(text=raw_text_7)
    return result


###
# Tests

def test_get_contiguous_chunks(extracted_text_persons, extracted_text_persons_explicit,
                               extracted_text_gpe, extracted_text_gpe_several):

    assert extracted_text_persons == [
        "Nancy Pelosi",
        "Jim Jordan"
    ]
    assert extracted_text_persons_explicit == [
        "Nancy Pelosi",
        "Jim Jordan"
    ]
    assert extracted_text_gpe == [
        "San Francisco",
    ]
    assert extracted_text_gpe_several == [
        "California", "Bakersfield", "Oakland", "Emeryville",
    ]


def test_get_full_names(extracted_full_names):
    assert extracted_full_names == [
        "Nancy Pelosi",
        "Jim Jordan"
    ]


def test_get_locations(extracted_get_locations):
    assert extracted_get_locations == [
        "San Francisco",
    ]


def test_get_organizations(extracted_get_organizations):
    assert extracted_get_organizations == [
        "Microsoft Corp",
    ]


def test_get_just_full_names1(extracted_just_names1):
    assert extracted_just_names1 == [
        "William J. Clinton",
        "Hillary Clinton",
    ]


def test_get_just_full_names2(extracted_just_names2):
    assert extracted_just_names2 == [
        "George W. Bush",
        "Dick Cheney",
    ]

def test_get_just_full_names3(extracted_just_names3):
    assert extracted_just_names3 == [
        "Andrew Ross Sorokin",
    ]

def test_is_proper_noun():
    extractor = textattack.misinformation.EntityExtraction()
    assert extractor.is_proper_noun(text="Uncle") == False
    assert extractor.is_proper_noun(text="sister") == False
    assert extractor.is_proper_noun(text="Brother") == False
    assert extractor.is_proper_noun(text="W.") == True
    assert extractor.is_proper_noun(text="Q,") == True
    assert extractor.is_proper_noun(text="Q?") == True


def test_get_full_names_params():
    extractor = textattack.misinformation.EntityExtraction()

    names = ["Billy Bob", "Thorton", "John Doe"]
    assert extractor.dedup_full_names(names=names, uniqueNames=True) == [
        "Billy Bob",
         "John Doe",
    ]

    names = ["Billy Bob", "Billy Bob Thorton", "John Doe"]
    assert extractor.dedup_full_names(names=names, uniqueNames=False) == [
        "Billy Bob",
        "Billy Bob Thorton",
         "John Doe",
    ]

    names = ['Joe Biden', 'Uncle', 'Robert Biden', 'Barack Obama', 'Father', 'Barack Obama Sr.', 'Joe Biden']
    assert extractor.dedup_full_names(names=names, uniqueNames=True) == [
        "Joe Biden",
        "Robert Biden",
        "Barack Obama", 
        "Barack Obama Sr.",
    ]


def test_get_stand_alone_names():
    extractor = textattack.misinformation.EntityExtraction()

    text = "His name is James Addison Baker and he lives here."
    assert extractor.get_unique_full_names(text) == [
        "James Addison Baker",
    ]

    text = "His name is James Addison Baker"
    assert extractor.get_unique_full_names(text) == [
        "James Addison Baker",
    ]

    text = "James Addison Baker and he lives here."
    assert extractor.get_unique_full_names(text) == [
        "James Addison Baker",
    ]

    text = "I believe James Addison Baker lives here and so does Bill Jones."
    assert extractor.get_unique_full_names(text) == [
        "James Addison Baker",
        "Bill Jones",
    ]

    text = "James Addison Baker lives here and so does Bill Jones."
    assert extractor.get_unique_full_names(text) == [
        "James Addison Baker",
        "Bill Jones",
    ]

    text = "His full name is James Addison Baker III but his friend has the plain name of Bill Jones."
    assert extractor.get_unique_full_names(text) == [
        "James Addison Baker III",
        "Bill Jones",
    ]


def test_get_stand_alone_org_locations():
    extractor = textattack.misinformation.EntityExtraction()

    text = "New York City"
    assert extractor.get_orgs_or_locations(text, "GPE") == [
        "New York City",
    ]

    text = "Chicago, IL"
    assert extractor.get_orgs_or_locations(text, "GPE") == [
        "Chicago",
        "IL",
    ]

    text = "Chicago, IL"
    assert extractor.get_locations(text) == [
        "Chicago",
        "IL",
    ]

    text = "Beijing is in China"
    assert extractor.get_locations(text) == [
        "Beijing",
        "China",
    ]

    text = "Microsoft Computer"
    assert extractor.get_orgs_or_locations(text, "ORGANIZATION") == [
        "Microsoft Computer",
    ]

    text = "The Senate"
    assert extractor.get_orgs_or_locations(text, "ORGANIZATION") == [
        "Senate",
    ]


