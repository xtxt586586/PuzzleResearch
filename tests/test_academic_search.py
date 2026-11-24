import unittest
from unittest.mock import patch

from common.config import settings
from tools.search.academic.arxiv_provider import ArxivProvider
from tools.search.academic.pubmed_provider import PubMedProvider
from tools.search.academic.semantic_scholar_provider import SemanticScholarProvider
from tools.search.manager import SearchManager


class FakeResponse:
    def __init__(self, payload=None, *, text="", status_code=200):
        self._payload = payload
        self.text = text
        self.status_code = status_code
        self.ok = status_code < 400

    def json(self):
        return self._payload


ARXIV_XML = """
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.00001</id>
    <title>Research title</title>
    <summary>Research abstract.</summary>
    <updated>2024-01-02T00:00:00Z</updated>
    <author><name>Researcher One</name></author>
    <category term="cs.AI" />
    <link title="pdf" href="https://arxiv.org/pdf/2401.00001" />
  </entry>
</feed>
"""


PUBMED_XML = """
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>123456</PMID>
      <Article>
        <ArticleTitle>Medical title</ArticleTitle>
        <Abstract><AbstractText Label="BACKGROUND">Medical abstract.</AbstractText></Abstract>
        <AuthorList><Author><LastName>Doe</LastName><ForeName>Jane</ForeName></Author></AuthorList>
        <Journal><Title>Medical Journal</Title></Journal>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


class AcademicSearchTests(unittest.TestCase):
    def test_availability_matches_local_configuration(self):
        with patch.object(settings, "arxiv_enabled", False), patch.object(
            settings, "pubmed_email", ""
        ):
            self.assertFalse(ArxivProvider().is_available())
            self.assertTrue(SemanticScholarProvider().is_available())
            self.assertFalse(PubMedProvider().is_available())

    def test_arxiv_parses_atom_response(self):
        with patch.object(settings, "arxiv_enabled", True), patch(
            "tools.search.academic.arxiv_provider.requests.get",
            return_value=FakeResponse(text=ARXIV_XML),
        ) as request:
            result = ArxivProvider().search("machine learning", max_results=2)

        self.assertEqual(result[0]["provider"], "arxiv")
        self.assertEqual(result[0]["url"], "https://arxiv.org/abs/2401.00001")
        self.assertIn("Researcher One", result[0]["content"])
        self.assertEqual(request.call_args.kwargs["params"]["max_results"], 2)

    def test_semantic_scholar_normalizes_paper(self):
        with patch.object(settings, "semantic_scholar_api_key", ""), patch(
            "tools.search.academic.semantic_scholar_provider.requests.get",
            return_value=FakeResponse(
                {
                    "data": [
                        {
                            "paperId": "paper-1",
                            "title": "Scholar title",
                            "abstract": "Scholar abstract.",
                            "year": 2024,
                            "citationCount": 12,
                            "authors": [{"name": "Scholar Author"}],
                        }
                    ]
                }
            ),
        ):
            result = SemanticScholarProvider().search("research", max_results=3)

        self.assertEqual(result[0]["provider"], "semantic_scholar")
        self.assertEqual(
            result[0]["url"],
            "https://www.semanticscholar.org/paper/paper-1",
        )
        self.assertIn("引用数：12", result[0]["content"])

    def test_pubmed_searches_then_fetches_articles(self):
        with patch.object(settings, "pubmed_email", "research@example.com"), patch.object(
            settings, "pubmed_api_key", ""
        ), patch(
            "tools.search.academic.pubmed_provider.requests.get",
            side_effect=[
                FakeResponse({"esearchresult": {"idlist": ["123456"]}}),
                FakeResponse(text=PUBMED_XML),
            ],
        ) as request:
            result = PubMedProvider().search("cancer", max_results=1)

        self.assertEqual(result[0]["provider"], "pubmed")
        self.assertEqual(result[0]["url"], "https://pubmed.ncbi.nlm.nih.gov/123456/")
        self.assertIn("Medical abstract.", result[0]["content"])
        self.assertEqual(request.call_count, 2)
        self.assertNotIn("research@example.com", str(result))

    def test_manager_registers_academic_providers(self):
        manager = SearchManager()
        self.assertEqual(
            {"arxiv", "semantic_scholar", "pubmed"}.intersection(manager.providers),
            {"arxiv", "semantic_scholar", "pubmed"},
        )


if __name__ == "__main__":
    unittest.main()
