"""Regression checks for source extraction, independent of live services."""
from pathlib import Path
import sys
import tempfile
import unittest
import json
from copy import deepcopy
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'fetch'))
from fetch_pois_osm import (fetch_pbf, campus_evidence, load_boundaries,
                            source_aliases, source_name_matches)


class PbfGeometryTests(unittest.TestCase):
    def extract(self, members):
        xml = '''<osm version="0.6">
          <node id="1" lat="30.53" lon="114.34"/>
          <node id="2" lat="30.55" lon="114.34"/>
          <node id="3" lat="30.54" lon="114.35"/>
          <node id="4" lat="30.53" lon="114.38"/>
          <way id="100"><nd ref="1"/><nd ref="2"/><nd ref="3"/></way>
          <way id="101"><nd ref="3"/><nd ref="4"/><nd ref="1"/></way>
          <relation id="200">''' + members + '''
            <tag k="type" v="multipolygon"/>
            <tag k="name" v="武汉大学实验楼"/>
            <tag k="building" v="university"/>
          </relation></osm>'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.osm'
            path.write_text(xml, encoding='utf-8')
            return fetch_pbf(path)

    def test_relation_uses_full_member_bounds_even_when_member_center_is_outside(self):
        rows = self.extract('<member type="way" ref="100" role="outer"/>'
                            '<member type="way" ref="101" role="outer"/>')
        relation = next(row for row in rows if row['type'] == 'relation')
        self.assertAlmostEqual(relation['center']['lon'], 114.36)
        self.assertAlmostEqual(relation['center']['lat'], 30.54)

    def test_relation_with_missing_member_is_not_given_partial_geometry(self):
        rows = self.extract('<member type="way" ref="101" role="outer"/>'
                            '<member type="way" ref="999" role="outer"/>')
        self.assertFalse(any(row['type'] == 'relation' for row in rows))


class CampusEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.boundaries = load_boundaries()

    def test_generic_dorm_at_sports_university_is_not_whu(self):
        campus, refs, reason = campus_evidence(114.370, 30.523,
            {'building': 'dormitory', 'name': '学生公寓10栋'}, self.boundaries)
        self.assertIsNone(campus)
        self.assertEqual(reason, 'other_education_area')
        self.assertIn('way/112940438', refs)

    def test_information_campus_uses_area_instead_of_wrong_nearest_anchor(self):
        campus, refs, reason = campus_evidence(114.353, 30.531,
            {'building': 'university', 'name': '测量仪器陈列馆'}, self.boundaries)
        self.assertEqual(campus, '信息学部')
        self.assertIn('relation/7728026', refs)

    def test_chengbo_gate_on_shared_institute_boundary_is_retained(self):
        campus, refs, reason = campus_evidence(114.3521343, 30.546276,
            {'barrier':'gate','name':'澄波门','foot':'permit'}, self.boundaries)
        self.assertEqual(campus, '工学部')
        self.assertEqual(reason, 'shared_boundary_gate')

    def test_aliases_split_osm_semicolons_without_inventing_equivalence(self):
        self.assertEqual(source_aliases({'alt_name': '甲;乙；丙', 'loc_name': '丁'}, '甲'), ['乙','丙','丁'])

    def test_same_label_at_distant_shop_does_not_hide_missing_branch(self):
        existing = [{'id':'shop_a','name':'自强超市','coordinates':{'lng':114.35,'lat':30.53}}]
        nearby, distant = source_name_matches(['自强超市'], existing, 114.36,30.54)
        self.assertFalse(nearby)
        self.assertIn('shop_a', distant)

    def test_close_unrelated_shop_does_not_become_an_alias(self):
        existing = [{'id':'shop_a','name':'一品豆花','coordinates':{'lng':114.35,'lat':30.53}}]
        nearby, distant = source_name_matches(['信息学部CBD'], existing, 114.35,30.53)
        self.assertFalse(nearby)
        self.assertFalse(distant)


class ReviewedProductionTests(unittest.TestCase):
    def test_reviewed_repairs_keep_types_campus_evidence_and_distinct_poi_names(self):
        root = Path(__file__).resolve().parents[1]
        pois = json.loads((root/'data/pois.json').read_text(encoding='utf-8'))['pois']
        by_id = {poi['id']: poi for poi in pois}
        for poi in pois:
            if any(ref.get('id') in {
                'way/1445906649', 'node/13266231641', 'node/9555974805',
                'way/306379133', 'way/1408203762', 'way/1445901385',
                'way/303368388', 'way/306232321', 'way/153752703',
            } for ref in poi.get('source_refs', [])):
                self.assertEqual(poi['type'], 'service', poi['name'])
        for poi_id in ('poi_395', 'poi_397', 'poi_415'):
            self.assertIn('relation/20349097', by_id[poi_id]['campus_boundary_refs'])
        self.assertNotIn('信息学部快递服务中心', by_id['poi_270']['aliases'])
        self.assertTrue(any(poi['name'] == '信息学部快递服务中心' for poi in pois))

    def test_review_application_is_idempotent_and_preserves_road_decisions(self):
        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / 'scripts' / 'validate'))
        from apply_reviewed_osm_pois import apply_review
        pois = json.loads((root/'data/pois.json').read_text(encoding='utf-8'))
        candidates = json.loads((root/'data/pois_osm_candidates.json').read_text(encoding='utf-8'))
        decisions = {'poi_decisions':[], 'road_decisions':[{'sentinel':'unchanged'}]}
        apply_review(pois, candidates, decisions)
        expected = deepcopy(pois)
        report = apply_review(pois, candidates, decisions)
        self.assertEqual(pois, expected)
        self.assertEqual(report['added_poi_ids'], [])
        self.assertEqual(report['aliases_added'], 0)
        self.assertEqual(decisions['road_decisions'], [{'sentinel':'unchanged'}])

    def test_osm_candidate_match_report_is_refreshed_against_the_current_master(self):
        root = Path(__file__).resolve().parents[1]
        pois = json.loads((root/'data/pois.json').read_text(encoding='utf-8'))['pois']
        candidates_doc = json.loads(
            (root/'data/pois_osm_candidates.json').read_text(encoding='utf-8'))
        candidates = {row['osm_id']: row for row in candidates_doc['candidates']}
        source_links = {
            ref['id']: poi['id']
            for poi in pois
            for ref in poi.get('source_refs', [])
            if ref.get('source') == 'OpenStreetMap' and ref.get('id') in candidates
        }

        for source_id, poi_id in source_links.items():
            with self.subTest(source_id=source_id):
                self.assertTrue(candidates[source_id]['already_exists'])
                self.assertIn(poi_id, candidates[source_id]['matched_poi_ids'])

        expected_missing = {
            row['osm_id'] for row in candidates_doc['candidates']
            if not row['already_exists']
        }
        reported_missing = {
            row['osm_id'] for row in candidates_doc['missing_from_production']
        }
        self.assertEqual(reported_missing, expected_missing)

    def test_reviewed_destinations_and_aliases_are_searchable(self):
        import spatial.poi as poi
        with patch.object(poi, '_record_search'):
            for query, expected in [('澄波门','澄波门'), ('明贤门','珞南二门'),
                                    ('工学部校医院','工学部社区卫生服务中心'),
                                    ('信息学部校医院','武汉大学医院二门诊部'),
                                    ('信息学部3舍','武汉大学信息学部学生宿舍3舍'),
                                    ('测量仪器陈列馆','测量仪器陈列馆')]:
                with self.subTest(query=query):
                    result = poi.find_poi(query)
                    self.assertIsNotNone(result)
                    self.assertEqual(result['name'], expected)

    def test_reviewed_sports_facilities_are_distinct_searchable_destinations(self):
        import spatial.poi as poi
        with patch.object(poi, '_record_search'):
            for name in ('星湖运动场', '竹园网球场', '露天羽毛球场', '大学生体育活动中心'):
                with self.subTest(name=name):
                    result = poi.find_poi(name)
                    self.assertIsNotNone(result)
                    self.assertEqual(result['name'], name)


if __name__ == '__main__':
    unittest.main()
