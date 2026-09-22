import unittest

from agent.core.principal import resolve_principal


class PrincipalTests(unittest.TestCase):
    def test_anonymous_principal_is_default(self):
        principal = resolve_principal()

        self.assertEqual(
            principal.user_id,
            "anonymous",
        )
        self.assertFalse(
            principal.authenticated,
        )

    def test_user_id_is_read_without_api_key(self):
        principal = resolve_principal(
            user_id="user-001",
        )

        self.assertEqual(
            principal.user_id,
            "user-001",
        )
        self.assertFalse(
            principal.authenticated,
        )

    def test_configured_api_key_authenticates_principal(self):
        principal = resolve_principal(
            user_id="user-001",
            api_key="secret",
            configured_api_key="secret",
        )

        self.assertEqual(
            principal.user_id,
            "user-001",
        )
        self.assertTrue(
            principal.authenticated,
        )

    def test_invalid_api_key_is_rejected(self):
        with self.assertRaises(PermissionError):
            resolve_principal(
                user_id="user-001",
                api_key="wrong",
                configured_api_key="secret",
            )


if __name__ == "__main__":
    unittest.main()
