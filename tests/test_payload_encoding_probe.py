import unittest
from tools.probe_jev_payload_encoding import pack_state, unpack_state, text


class PayloadEncodingProbeTests(unittest.TestCase):
    def test_references_and_tables_restore_exact_numbers_types_and_order(self):
        path = [{'x': i * 0.12345678901234567, 'y': -0.0} for i in range(30)]
        state = {'path_a': path, 'path_b': path,
                 'rows': [{'a': None, 'b': True}, {'a': 0, 'b': False}, {'a': 0.0, 'b': True}],
                 'text': '导航', 'empty_list': [], 'empty_dict': {}}
        packed = pack_state(state)
        self.assertGreater(len(packed['definitions']), 0)
        self.assertEqual(text(unpack_state(packed)), text(state))
        self.assertEqual(len(state['path_a']), 30)

    def test_mixed_rows_and_missing_fields_remain_distinct(self):
        state = {'mixed': [{'a': None}, {}, {'a': 0, 'b': 2}, [1, 2], 'x']}
        for dedup in (True, False):
            self.assertEqual(text(unpack_state(pack_state(state, dedup))), text(state))

    def test_reserved_markers_rejected_instead_of_corrupting_input(self):
        for marker in ('__jev_ref', '__jev_table'):
            with self.assertRaises(ValueError):
                pack_state({'nested': [{marker: 'source data'}]})

if __name__ == '__main__':
    unittest.main()
