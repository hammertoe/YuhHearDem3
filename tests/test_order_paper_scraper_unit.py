from lib.order_papers.scraper import parse_order_paper_search_html


def test_parse_order_paper_search_html_extracts_primary_and_attachments():
    html = """
    <table class="table table-hover table-responsive">
        <tr class="modern-style">
            <td width="70%">
                <a href="https://www.barbadosparliament.com/uploads/sittings/attachments/main.pdf">
                    3rd Sitting - 2nd March 2026
                </a>
                <div style="padding: 4px 15px 5px 30px">
                    <div style="padding-top: 6px;">
                        &middot;&nbsp;&nbsp;
                        <a href="https://www.barbadosparliament.com/uploads/sittings/attachments/booklet.pdf">
                            Booklet
                        </a>
                    </div>
                    <div style="padding-top: 6px;">
                        &middot;&nbsp;&nbsp;
                        <a href="https://www.barbadosparliament.com/uploads/sittings/attachments/supplement.pdf">
                            Supplement to Order Paper
                        </a>
                    </div>
                </div>
            </td>
            <td>2026-03-02</td>
        </tr>
    </table>
    """

    entries = parse_order_paper_search_html(html, chamber="house")

    assert len(entries) == 1
    entry = entries[0]
    assert entry.title == "3rd Sitting - 2nd March 2026"
    assert entry.posted_date == "2026-03-02"
    assert (
        entry.pdf_url == "https://www.barbadosparliament.com/uploads/sittings/attachments/main.pdf"
    )
    assert len(entry.attachments) == 2
    assert entry.attachments[0].label == "Booklet"
    assert (
        entry.attachments[0].url
        == "https://www.barbadosparliament.com/uploads/sittings/attachments/booklet.pdf"
    )
    assert entry.attachments[1].label == "Supplement to Order Paper"
    assert (
        entry.attachments[1].url
        == "https://www.barbadosparliament.com/uploads/sittings/attachments/supplement.pdf"
    )
    assert entry.chamber == "house"
